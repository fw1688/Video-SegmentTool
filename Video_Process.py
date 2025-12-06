import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict
import logging
from collections import deque
import time
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_segmentation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GameVideoSegmenter:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._check_gpu_status()
        self.setup_models()
        self.setup_transforms()

        # 极大降低阈值，提高好帧采样率
        self.motion_threshold = 0.01  # 大幅降低运动阈值
        self.brightness_threshold = 15  # 大幅降低亮度阈值
        self.sharpness_threshold = 10  # 大幅降低清晰度阈值
        self.mouse_movement_threshold = 0.3  # 提高鼠标移动阈值，更宽容

        # 分段参数
        self.min_segment_duration = 1.0  # 最小分段时长(秒)
        self.max_gap_duration = 0.5  # 最大容忍间隔(秒)
        self.stable_frames_threshold = 10  # 稳定帧数阈值

        # 状态变量
        self.frame_buffer = deque(maxlen=5)
        self.good_frames_count = 0
        self.bad_frames_count = 0
        self.current_segment_start = None

    def _check_gpu_status(self):
        """检查GPU状态和性能"""
        logger.info("=== GPU/CPU 设备状态检查 ===")

        if torch.cuda.is_available():
            logger.info(f"✅ CUDA可用，使用设备: {self.device}")
            gpu_count = torch.cuda.device_count()
            logger.info(f"📊 检测到 {gpu_count} 个GPU设备")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"  GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f}GB")
        else:
            logger.warning("❌ CUDA不可用，使用CPU进行计算")

        logger.info("=== 设备检查完成 ===\n")

    def setup_models(self):
        """设置需要的模型"""
        logger.info("正在加载AI模型...")
        try:
            # 加载目标检测模型用于UI元素检测
            self.detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            logger.info("✅ YOLOv5模型加载完成")
        except Exception as e:
            logger.warning(f"YOLOv5加载失败: {e}")
            self.detection_model = None

    def setup_transforms(self):
        """设置图像变换"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def calculate_brightness(self, frame: np.ndarray) -> float:
        """计算帧的亮度"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return np.mean(gray)

    def calculate_sharpness(self, frame: np.ndarray) -> float:
        """计算帧的清晰度（使用拉普拉斯方差）"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_motion(self, current_frame: np.ndarray) -> float:
        """计算与前一帧的运动量"""
        if not self.frame_buffer:
            self.frame_buffer.append(current_frame)
            return 0.0

        # 与缓冲区中的前一帧比较
        prev_frame = self.frame_buffer[-1]

        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 计算光流或帧差异
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_score = np.mean(diff) / 255.0

        self.frame_buffer.append(current_frame)
        return motion_score

    def detect_ui_elements(self, frame: np.ndarray) -> bool:
        """检测UI元素 - 简化版本，只检测显著UI"""
        if self.detection_model is None:
            return False

        try:
            results = self.detection_model(frame)
            detections = results.xyxy[0].cpu().numpy()

            h, w = frame.shape[:2]
            ui_count = 0

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.7:  # 提高置信度阈值
                    # 检查是否在边缘区域且尺寸较大
                    if (x1 < 0.1 * w or x2 > 0.9 * w or y1 < 0.1 * h or y2 > 0.9 * h):
                        area = (x2 - x1) * (y2 - y1)
                        if area > (w * h * 0.1):  # 面积大于画面10%
                            ui_count += 1
                            if ui_count >= 1:
                                return True
        except Exception as e:
            logger.warning(f"UI检测失败: {e}")

        return False

    def is_good_frame(self, frame: np.ndarray, frame_count: int) -> Tuple[bool, List[str]]:
        """判断帧是否为好帧 - 极大放宽条件"""
        reasons = []
        is_good = True

        # 1. 检查亮度 - 极大放宽
        brightness = self.calculate_brightness(frame)
        if brightness < self.brightness_threshold:
            reasons.append(f'亮度不足: {brightness:.1f}')
            is_good = False

        # 2. 检查清晰度 - 极大放宽
        sharpness = self.calculate_sharpness(frame)
        if sharpness < self.sharpness_threshold:
            reasons.append(f'清晰度不足: {sharpness:.1f}')
            is_good = False

        # 3. 检查运动 - 极大放宽
        motion = self.calculate_motion(frame)
        if motion < self.motion_threshold and frame_count > 0:
            reasons.append(f'运动不足: {motion:.4f}')
            is_good = False

        # 4. 检查UI元素 - 只检测显著UI
        has_ui = self.detect_ui_elements(frame)
        if has_ui:
            reasons.append('检测到显著UI')
            is_good = False

        return is_good, reasons

    def frame_to_time(self, frame_number: int, fps: float) -> str:
        """将帧数转换为时间格式"""
        total_seconds = frame_number / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def log_segment_details(self, segment: Dict, fps: float, segment_id: int, end_reason: str):
        """记录分段详细信息"""
        start_time_str = self.frame_to_time(segment['start_frame'], fps)
        end_time_str = self.frame_to_time(segment['end_frame'], fps)

        logger.info("🎬" + "=" * 60)
        logger.info(f"📊 分段 #{segment_id} 详细信息:")
        logger.info(f"  开始帧: {segment['start_frame']}")
        logger.info(f"  结束帧: {segment['end_frame']}")
        logger.info(f"  总帧数: {segment['frame_count']}")
        logger.info(f"  开始时间: {start_time_str}")
        logger.info(f"  结束时间: {end_time_str}")
        logger.info(f"  持续时间: {segment['duration']:.2f}秒")
        logger.info(f"  分段原因: {end_reason}")
        logger.info("=" * 60)

    def segment_video(self, video_path: str, output_dir: str,
                      skip_frames: int = 0, max_frames: int = None) -> Dict:
        """分段处理视频 - 按连续好帧进行分段"""
        # 性能监控
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info("📹" + "=" * 60)
        logger.info(f"开始分段视频: {video_path}")
        logger.info(f"原始分辨率: {width}x{height}")
        logger.info(f"帧率: {fps} FPS")
        logger.info(f"总帧数: {total_frames}")
        logger.info(f"分段参数: 最小分段{self.min_segment_duration}秒, 最大间隔{self.max_gap_duration}秒")
        logger.info("=" * 60)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化分段状态
        segments = []  # 存储分段信息: (start_frame, end_frame, frame_count)
        current_segment_frames = []
        gap_frames = 0
        frame_count = 0
        processed_frames = 0

        # 统计信息
        good_frames_total = 0
        bad_frames_total = 0
        rejection_reasons = {}

        # 分段统计
        segment_id = 0
        segment_start_time = None

        # 重置状态
        self.frame_buffer.clear()
        self.good_frames_count = 0
        self.bad_frames_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count < skip_frames:
                frame_count += 1
                continue

            if max_frames and processed_frames >= max_frames:
                break

            # 判断帧质量
            is_good, reasons = self.is_good_frame(frame, frame_count)
            processed_frames += 1

            if is_good:
                # 好帧
                self.good_frames_count += 1
                good_frames_total += 1

                # 重置坏帧计数
                gap_frames = 0

                # 如果是新分段开始
                if not current_segment_frames:
                    segment_start_time = time.time()
                    logger.info(f"🎬 开始新分段于帧 {frame_count} (时间: {self.frame_to_time(frame_count, fps)})")

                # 添加到当前分段
                current_segment_frames.append(frame_count)

            else:
                # 坏帧
                self.bad_frames_count += 1
                bad_frames_total += 1

                # 记录拒绝原因
                for reason in reasons:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

                # 如果当前有活跃分段，增加间隔计数
                if current_segment_frames:
                    gap_frames += 1

                    # 如果间隔超过阈值，结束当前分段
                    if gap_frames / fps > self.max_gap_duration:
                        # 检查分段长度是否足够
                        segment_duration = len(current_segment_frames) / fps
                        if segment_duration >= self.min_segment_duration:
                            segment_id += 1
                            segment_info = {
                                'start_frame': current_segment_frames[0],
                                'end_frame': current_segment_frames[-1],
                                'frame_count': len(current_segment_frames),
                                'duration': segment_duration,
                                'segment_id': segment_id
                            }
                            segments.append(segment_info)

                            # 记录分段详细信息
                            self.log_segment_details(segment_info, fps, segment_id, "连续坏帧超阈值")

                        else:
                            segment_processing_time = time.time() - segment_start_time
                            logger.info(
                                f"❌ 分段过短丢弃: {segment_duration:.1f}秒, 处理时间: {segment_processing_time:.1f}秒")

                        # 重置分段
                        current_segment_frames = []
                        gap_frames = 0

            frame_count += 1

            if processed_frames % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_processed = processed_frames / elapsed_time
                logger.info(f"⏱️ 进度: {processed_frames}/{max_frames} 帧 "
                            f"({processed_frames / total_frames * 100:.1f}%) | "
                            f"好帧: {good_frames_total} | 坏帧: {bad_frames_total} | "
                            f"分段: {len(segments)} | 速度: {fps_processed:.1f} FPS")

        # 处理最后一个分段
        if current_segment_frames:
            segment_duration = len(current_segment_frames) / fps
            if segment_duration >= self.min_segment_duration:
                segment_id += 1
                segment_info = {
                    'start_frame': current_segment_frames[0],
                    'end_frame': current_segment_frames[-1],
                    'frame_count': len(current_segment_frames),
                    'duration': segment_duration,
                    'segment_id': segment_id
                }
                segments.append(segment_info)

                # 记录分段详细信息
                self.log_segment_details(segment_info, fps, segment_id, "视频结束")
            else:
                segment_processing_time = time.time() - segment_start_time
                logger.info(f"❌ 最终分段过短丢弃: {segment_duration:.1f}秒, 处理时间: {segment_processing_time:.1f}秒")

        cap.release()

        # 提取分段视频
        self._extract_segments(video_path, output_dir, segments, fps, width, height)

        # 构建结果摘要
        total_time = time.time() - start_time

        results_summary = {
            'video_info': {
                'path': video_path,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'total_frames': total_frames
            },
            'processing_stats': {
                'processed_frames': processed_frames,
                'good_frames': good_frames_total,
                'bad_frames': bad_frames_total,
                'segments_count': len(segments),
                'processing_time': total_time,
                'average_fps': processed_frames / total_time if total_time > 0 else 0,
                'device_used': 'GPU' if torch.cuda.is_available() else 'CPU',
                'good_frame_ratio': good_frames_total / max(processed_frames, 1) * 100
            },
            'segments': segments,
            'rejection_reasons': rejection_reasons
        }

        # 保存结果摘要
        summary_path = os.path.join(output_dir, 'segmentation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        # 最终统计日志
        logger.info("🎯" + "=" * 60)
        logger.info("分段处理完成!")
        logger.info(f"📊 最终统计:")
        logger.info(f"  总处理帧数: {processed_frames}")
        logger.info(f"  好帧数量: {good_frames_total}")
        logger.info(f"  坏帧数量: {bad_frames_total}")
        logger.info(f"  好帧比例: {results_summary['processing_stats']['good_frame_ratio']:.1f}%")
        logger.info(f"  有效分段: {len(segments)}")
        logger.info(f"  总处理时间: {total_time:.1f}秒")
        logger.info(f"  平均处理速度: {results_summary['processing_stats']['average_fps']:.1f} FPS")
        logger.info("=" * 60)

        return results_summary

    def _extract_segments(self, video_path: str, output_dir: str,
                          segments: List[Dict], fps: float,
                          width: int, height: int):
        """提取分段视频"""
        logger.info("💾 开始提取分段视频...")

        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"segment_{segment['segment_id']:03d}.mp4")

            # 设置视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 读取并写入分段
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start_frame'])

            frames_written = 0
            segment_extract_start = time.time()

            while frames_written < segment['frame_count']:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1

            cap.release()
            out.release()

            extract_time = time.time() - segment_extract_start

            logger.info(f"📹 分段 {segment['segment_id']} 已保存: {output_path}")
            logger.info(f"   帧范围: {segment['start_frame']}-{segment['end_frame']}")
            logger.info(f"   帧数量: {frames_written}")
            logger.info(f"   持续时间: {segment['duration']:.2f}秒")
            logger.info(f"   提取时间: {extract_time:.2f}秒")


def main():
    """主函数"""
    # 记录程序开始时间
    program_start = time.time()
    logger.info("🚀 视频分段程序启动")
    logger.info(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    segmenter = GameVideoSegmenter()

    # 配置参数
    video_path = "2025-11-22 09-04-51.mp4"  # 输入视频路径
    output_dir = "video_segments"  # 输出目录
    skip_frames = 0  # 跳过的帧数
    max_frames = 10800  # 最大处理帧数

    # 分段视频
    results = segmenter.segment_video(
        video_path=video_path,
        output_dir=output_dir,
        skip_frames=skip_frames,
        max_frames=max_frames
    )

    # 打印摘要
    program_duration = time.time() - program_start

    print("\n" + "=" * 60)
    print("视频分段结果摘要")
    print("=" * 60)
    print(f"输入视频: {results['video_info']['path']}")
    print(f"视频信息: {results['video_info']['resolution']}, {results['video_info']['fps']} FPS")
    print(f"使用设备: {results['processing_stats']['device_used']}")
    print(f"处理帧数: {results['processing_stats']['processed_frames']}")
    print(f"好帧数量: {results['processing_stats']['good_frames']}")
    print(f"坏帧数量: {results['processing_stats']['bad_frames']}")
    print(f"好帧比例: {results['processing_stats']['good_frame_ratio']:.1f}%")
    print(f"分段数量: {results['processing_stats']['segments_count']}")
    print(f"处理时间: {results['processing_stats']['processing_time']:.1f}秒")
    print(f"处理速度: {results['processing_stats']['average_fps']:.1f} FPS")
    print(f"程序总运行时间: {program_duration:.1f}秒")

    print(f"\n分段详情:")
    for i, segment in enumerate(results['segments']):
        start_time = segment['start_frame'] / results['video_info']['fps']
        end_time = segment['end_frame'] / results['video_info']['fps']
        print(f"  🎬 分段{segment['segment_id']}: "
              f"帧{segment['start_frame']}-{segment['end_frame']} "
              f"({segment['frame_count']}帧, {segment['duration']:.1f}秒) "
              f"时间[{start_time:.1f}s-{end_time:.1f}s]")

    if results['rejection_reasons']:
        print(f"\n拒绝原因统计:")
        for reason, count in sorted(results['rejection_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  ❌ {reason}: {count}次")

    print("=" * 60)

    # 记录程序结束
    logger.info(f"🏁 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏱️ 程序总运行时间: {program_duration:.1f}秒")


if __name__ == "__main__":
    main()