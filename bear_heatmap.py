import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import json
import csv
import time
from datetime import timedelta
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path

# Constants
DEFAULT_SKIP_SECONDS = 0.5
ZOOM_SIZE = 200
ZOOM_SCALE = 3
FOX_BLOB_WIDTH = 80
FOX_BLOB_HEIGHT = 50
JUMP_THRESHOLD = 300  # pixels - adjust based on your video


@dataclass
class Annotation:
    """Single annotation point"""

    frame_index: int
    x: Optional[int]
    y: Optional[int]
    timestamp: float
    confidence: float = 1.0


class VideoAnnotator:
    """Main video annotation class"""

    def __init__(self, video_path: str, skip_seconds: float, output_prefix: str):
        self.video_path = video_path
        self.skip_seconds = skip_seconds
        self.output_prefix = output_prefix

        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_minutes = self.total_frames / self.fps / 60
        self.skip_frames = int(self.fps * skip_seconds)

        # Annotation state
        self.annotations: List[Annotation] = []
        self.current_frame = 0
        self.last_position = None
        self.start_time = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.show_zoom = True
        self.show_path = True
        self.review_mode = False

        # Colors for visualization
        self.colors = self._create_colormap()

    def _create_colormap(self):
        """Create blue-to-red colormap"""
        colors = [
            "#000033",
            "#000055",
            "#0000FF",
            "#0066FF",
            "#00CCFF",
            "#66FFCC",
            "#FFFF66",
            "#FFCC00",
            "#FF6600",
            "#FF0000",
            "#CC0000",
        ]
        return LinearSegmentedColormap.from_list("fox_heatmap", colors, N=100)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x, self.mouse_y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self._add_annotation(x, y)
            self._check_jump_detection(x, y)
            self._next_frame()

        elif event == cv2.EVENT_RBUTTONDOWN and self.last_position:
            self._add_annotation(self.last_position[0], self.last_position[1])
            self._next_frame()

    def _add_annotation(self, x: int, y: int):
        """Add annotation point"""
        if self.start_time is None:
            self.start_time = time.time()

        annotation = Annotation(
            frame_index=self.current_frame,
            x=x,
            y=y,
            timestamp=self.current_frame / self.fps,
        )
        self.annotations.append(annotation)
        self.last_position = (x, y)

        elapsed = time.time() - self.start_time
        print(f"✓ Frame {self.current_frame}: ({x}, {y}) | Time: {elapsed:.1f}s")

    def _check_jump_detection(self, x: int, y: int):
        """Detect suspicious jumps in position"""
        if self.last_position:
            distance = np.sqrt(
                (x - self.last_position[0]) ** 2 + (y - self.last_position[1]) ** 2
            )
            if distance > JUMP_THRESHOLD:
                print(f"⚠️  WARNING: Large jump detected ({distance:.0f} pixels)")
                print("   Press 'U' to undo if this was a mistake")

    def _next_frame(self):
        """Advance to next frame"""
        if not self.review_mode:
            self.current_frame = self.current_frame + self.skip_frames

        self._show_frame()

    def _prev_frame(self):
        """Go to previous frame"""
        if self.annotations:
            self.annotations.pop()
            self.current_frame = max(0, self.current_frame - self.skip_frames)
            if self.annotations:
                last = self.annotations[-1]
                self.last_position = (last.x, last.y) if last.x else None
            print("↶ Went back one frame")
        self._show_frame()

    def _quick_review(self):
        """Enter/exit quick review mode"""
        self.review_mode = not self.review_mode
        if self.review_mode:
            self.review_start_frame = max(0, self.current_frame - 10 * self.skip_frames)
            self.current_frame = self.review_start_frame
            print("🔍 REVIEW MODE: Showing last 10 frames. Press 'R' again to exit")
        else:
            self.current_frame = min(
                self.current_frame + 10 * self.skip_frames, self.total_frames - 1
            )
            print("✓ Review complete, continuing annotation")
        self._show_frame()

    def _create_zoom_window(self, frame):
        """Create magnified view around cursor"""
        zoom_half = ZOOM_SIZE // 2
        x1 = max(0, self.mouse_x - zoom_half // ZOOM_SCALE)
        y1 = max(0, self.mouse_y - zoom_half // ZOOM_SCALE)
        x2 = min(self.width, x1 + ZOOM_SIZE // ZOOM_SCALE)
        y2 = min(self.height, y1 + ZOOM_SIZE // ZOOM_SCALE)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            zoom = cv2.resize(
                roi, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_LINEAR
            )

            # Add crosshair to zoom
            cv2.line(
                zoom, (ZOOM_SIZE // 2, 0), (ZOOM_SIZE // 2, ZOOM_SIZE), (0, 255, 0), 1
            )
            cv2.line(
                zoom, (0, ZOOM_SIZE // 2), (ZOOM_SIZE, ZOOM_SIZE // 2), (0, 255, 0), 1
            )

            return zoom
        return None

    def _draw_movement_path(self, frame):
        """Draw connected path of fox movement"""
        valid_annotations = [(a.x, a.y) for a in self.annotations[-30:] if a.x]

        if len(valid_annotations) > 1:
            points = np.array(valid_annotations, np.int32)
            # Draw path with gradient
            for i in range(1, len(points)):
                alpha = i / len(points)
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                thickness = int(1 + 2 * alpha)
                cv2.line(
                    frame, tuple(points[i - 1]), tuple(points[i]), color, thickness
                )

            # Draw points
            for i, point in enumerate(points):
                alpha = (i + 1) / len(points)
                cv2.circle(frame, tuple(point), 3, (0, int(255 * alpha), 0), -1)

    def _draw_legend(self, frame):
        """Draw instruction legend on frame"""
        h, w = frame.shape[:2]
        legend_height = 140
        overlay = frame.copy()

        # Semi-transparent background
        cv2.rectangle(overlay, (10, h - legend_height), (350, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Legend text
        instructions = [
            "CONTROLS:",
            "LEFT CLICK: Mark fox position",
            "RIGHT CLICK: Reuse last position",
            "SPACE: Fox not visible",
            "G: Go to timestamp",
            "X: Blackout/Recording break",
            "R: Quick review (last 10 frames)",
            "Z: Toggle zoom window",
            "P: Toggle path display",
            "B: Go back | U: Undo | Q: Finish",
        ]

        y_offset = h - legend_height + 20
        for instruction in instructions:
            cv2.putText(
                frame,
                instruction,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
            y_offset += 15

        return frame

    def _show_frame(self):
        """Display current frame with overlays"""
        if self.current_frame >= self.total_frames:
            self.finish()
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw movement path
        if self.show_path:
            self._draw_movement_path(display)

        # Draw crosshair
        cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 1, cv2.LINE_AA)

        # Show last position
        if self.last_position:
            cv2.circle(display, self.last_position, 20, (0, 255, 255), 2)
            cv2.putText(
                display,
                "Last",
                (self.last_position[0] + 25, self.last_position[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        # Progress bar
        progress = (self.current_frame / self.total_frames) * 100
        bar_length = int((w - 20) * (self.current_frame / self.total_frames))
        cv2.rectangle(display, (10, 10), (w - 10, 30), (50, 50, 50), -1)
        cv2.rectangle(display, (10, 10), (10 + bar_length, 30), (0, 255, 0), -1)

        # Status text
        time_in_video = self.current_frame / self.fps
        status = "REVIEW MODE" if self.review_mode else "ANNOTATING"

        if self.start_time:
            elapsed = str(timedelta(seconds=int(time.time() - self.start_time)))
            timer_text = f" | Timer: {elapsed}"
        else:
            timer_text = " | Timer: Ready"

        cv2.putText(
            display,
            f"{status} | {progress:.1f}% | Frame {self.current_frame} | "
            f"Time: {time_in_video:.1f}s | Clicks: {len(self.annotations)}{timer_text}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw legend
        display = self._draw_legend(display)

        # Show zoom window
        if self.show_zoom:
            zoom = self._create_zoom_window(frame)
            if zoom is not None:
                display[10 : 10 + ZOOM_SIZE, w - ZOOM_SIZE - 10 : w - 10] = zoom
                cv2.rectangle(
                    display,
                    (w - ZOOM_SIZE - 11, 9),
                    (w - 9, 11 + ZOOM_SIZE),
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    display,
                    f"Zoom {ZOOM_SCALE}x",
                    (w - ZOOM_SIZE - 10, ZOOM_SIZE + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

        cv2.imshow("Fox Tracker", display)

    def run_annotation(self):
        """Main annotation loop"""
        cv2.namedWindow("Fox Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fox Tracker", 1400, 900)
        cv2.setMouseCallback("Fox Tracker", self.mouse_callback)

        self._show_frame()

        while True:
            if self.current_frame >= self.total_frames:
                print("\n📹 Reached end of video!")
                self.finish()
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                self.finish()
                break
            elif key == 32:  # SPACE - not visible
                if self.start_time is None:
                    self.start_time = time.time()
                self.annotations.append(
                    Annotation(
                        frame_index=self.current_frame,
                        x=None,
                        y=None,
                        timestamp=self.current_frame / self.fps,
                    )
                )
                print("○ Fox not visible")
                self._next_frame()
            elif key == ord("s"):  # S - Skip frame
                if self.start_time is None:
                    self.start_time = time.time()

                self._next_frame()
            elif key == ord("b"):  # B - back
                self._prev_frame()
            elif key == ord("u"):  # U - undo
                self._prev_frame()
            elif key == ord("r"):  # R - review
                self._quick_review()
            elif key == ord("z"):  # Z - toggle zoom
                self.show_zoom = not self.show_zoom
                print(f"Zoom: {'ON' if self.show_zoom else 'OFF'}")
            elif key == ord("p"):  # P - toggle path
                self.show_path = not self.show_path
                print(f"Path: {'ON' if self.show_path else 'OFF'}")
            elif key == ord("n") and self.review_mode:  # N - next in review
                self.current_frame = min(
                    self.current_frame + self.skip_frames,
                    self.review_start_frame + 10 * self.skip_frames,
                )
                self._show_frame()
            elif key == ord("x"):  # X for blackout/break
                if self.start_time is None:
                    self.start_time = time.time()

                self.annotations.append(
                    Annotation(
                        frame_index=self.current_frame,
                        x=-1,  # Special marker for "break in recording"
                        y=-1,
                        timestamp=self.current_frame / self.fps,
                    )
                )
                print("⬛ Recording break - path will be disconnected")
                self._next_frame()
            elif key == ord("g"):  # G - Go to specific timestamp
                if self.start_time is None:
                    self.start_time = time.time()

                # Get input from terminal
                try:
                    time_str = input("\nJump to time (MM:SS or seconds): ").strip()

                    if not time_str:
                        print("Jump cancelled")
                        continue

                    # Parse time input
                    if ":" in time_str:
                        parts = time_str.split(":")
                        if len(parts) == 2:
                            seconds = int(parts[0]) * 60 + float(parts[1])
                        elif len(parts) == 3:  # HH:MM:SS format
                            seconds = (
                                int(parts[0]) * 3600
                                + int(parts[1]) * 60
                                + float(parts[2])
                            )
                    else:
                        seconds = float(time_str)

                    # Calculate target frame
                    target_frame = int(self.fps * seconds)

                    # Validate and jump
                    if target_frame < 0:
                        print("❌ Invalid time (negative)")
                    elif target_frame >= self.total_frames:
                        print(
                            f"❌ Time exceeds video length ({self.duration_minutes:.1f} min)"
                        )
                    else:
                        self.current_frame = target_frame
                        time_formatted = str(timedelta(seconds=int(seconds)))
                        print(
                            f"✅ Jumped to {time_formatted} (frame {self.current_frame})"
                        )

                        # Optional: Add a note in annotations about the jump
                        if self.annotations and len(self.annotations) > 0:
                            print(
                                "   Note: You may want to press 'X' to mark discontinuity"
                            )

                    self._show_frame()

                except ValueError:
                    print(
                        "❌ Invalid time format. Use MM:SS or seconds (e.g., 2:30 or 150)"
                    )
                except Exception as e:
                    print(f"❌ Error: {e}")

    def export_csv(self):
        """Export annotations to CSV"""
        csv_file = f"{self.output_prefix}_annotations.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "x", "y", "timestamp_seconds", "type"])
            for ann in self.annotations:
                if ann.x == -1:
                    ann_type = "blackout"
                elif ann.x is None:
                    ann_type = "not_visible"
                else:
                    ann_type = "visible"
                writer.writerow(
                    [ann.frame_index, ann.x or "", ann.y or "", ann.timestamp, ann_type]
                )
        print(f"✅ Exported annotations to {csv_file}")
        return csv_file

    def export_json(self):
        """Export annotations and metadata to JSON"""
        json_file = f"{self.output_prefix}_data.json"
        data = {
            "video_path": self.video_path,
            "video_properties": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "duration_minutes": self.duration_minutes,
            },
            "annotation_settings": {
                "skip_seconds": self.skip_seconds,
                "skip_frames": self.skip_frames,
            },
            "annotations": [asdict(ann) for ann in self.annotations],
            "statistics": self._calculate_statistics(),
        }

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Exported data to {json_file}")
        return json_file

    def _calculate_statistics(self) -> Dict:
        """Calculate annotation statistics"""
        valid_annotations = [a for a in self.annotations if a.x and a.x > 0]
        blackout_frames = [a for a in self.annotations if a.x == -1]
        not_visible_frames = [a for a in self.annotations if a.x is None]

        if not valid_annotations:
            return {}

        x_coords = [a.x for a in valid_annotations]
        y_coords = [a.y for a in valid_annotations]

        # Calculate total distance (breaking at blackouts)
        total_distance = 0
        for i in range(1, len(self.annotations)):
            curr = self.annotations[i]
            prev = self.annotations[i - 1]

            # Only calculate distance between valid consecutive positions
            if curr.x and curr.x > 0 and prev.x and prev.x > 0:
                dx = curr.x - prev.x
                dy = curr.y - prev.y
                total_distance += np.sqrt(dx**2 + dy**2)

        # Calculate visibility excluding blackouts
        total_real_frames = len(self.annotations) - len(blackout_frames)

        return {
            "total_annotations": len(self.annotations),
            "visible_annotations": len(valid_annotations),
            "not_visible_frames": len(not_visible_frames),
            "blackout_frames": len(blackout_frames),
            "real_frames_analyzed": total_real_frames,
            "visibility_percentage": (
                100 * len(valid_annotations) / total_real_frames
                if total_real_frames > 0
                else 0
            ),
            "center_of_activity": {
                "x": np.mean(x_coords),
                "y": np.mean(y_coords),
                "std_x": np.std(x_coords),
                "std_y": np.std(y_coords),
            },
            "total_distance_pixels": total_distance,
            "annotation_time_seconds": (
                time.time() - self.start_time if self.start_time else 0
            ),
        }

    def finish(self):
        """Complete annotation and generate outputs"""
        cv2.destroyAllWindows()

        if not self.annotations:
            print("No annotations to save")
            return

        total_time = time.time() - self.start_time if self.start_time else 0

        print(f"\n{'='*50}")
        print(f"✅ ANNOTATION COMPLETE")
        print(f"{'='*50}")
        print(f"⏱️  Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"📊 Total annotations: {len(self.annotations)}")
        print(f"🎯 Speed: {total_time/len(self.annotations):.1f}s per annotation")
        print(f"{'='*50}\n")

        # Export data
        self.export_csv()
        self.export_json()

        # Generate visualizations
        visualizer = HeatmapVisualizer(self)
        visualizer.create_heatmap()
        # self.create_animated_gif()

        print(f"\n✅ All files saved with prefix: {self.output_prefix}")

        if self.cap:
            self.cap.release()


class HeatmapVisualizer:
    """Handles heatmap generation and visualization"""

    def __init__(self, annotator: VideoAnnotator):
        self.annotator = annotator

    def _create_fox_blob(self, x: int, y: int) -> np.ndarray:
        """Create fox-shaped gaussian blob"""
        Y, X = np.ogrid[: self.annotator.height, : self.annotator.width]

        dist_x = (X - x) / FOX_BLOB_WIDTH
        dist_y = (Y - y) / FOX_BLOB_HEIGHT
        dist_sq = dist_x**2 + dist_y**2

        gaussian = np.exp(-dist_sq / 2)
        center_boost = np.exp(-dist_sq / 0.5) * 0.3

        return gaussian + center_boost

    def create_heatmap(self):
        """Generate and save heatmap visualizations"""
        print("Generating heatmap visualizations...")

        # Create heatmap
        heatmap = np.zeros(
            (self.annotator.height, self.annotator.width), dtype=np.float32
        )

        valid_annotations = [a for a in self.annotator.annotations if a.x and a.x > 0]

        for ann in valid_annotations:
            blob = self._create_fox_blob(ann.x, ann.y)
            heatmap += blob

        # Smooth and normalize
        heatmap = gaussian_filter(heatmap, sigma=15)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Get background frame
        self.annotator.cap.set(
            cv2.CAP_PROP_POS_FRAMES, self.annotator.total_frames // 2
        )
        ret, bg_frame = self.annotator.cap.read()

        if ret and bg_frame is not None:
            bg_frame_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
        else:
            # Create a gray background if video frame unavailable
            bg_frame_rgb = (
                np.ones(
                    (self.annotator.height, self.annotator.width, 3), dtype=np.uint8
                )
                * 128
            )
            print(
                "Note: Could not read video frame for background, using gray background"
            )

        print("Creating individual plot files...")

        # 1. Heatmap Overlay
        fig1 = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.imshow(bg_frame_rgb)
        # Apply log scaling to make low-activity areas visible
        heatmap_log = np.log1p(heatmap * 100)  # log1p handles zeros safely
        if heatmap_log.max() > 0:
            heatmap_log = heatmap_log / heatmap_log.max()

        # Then use heatmap_log for visualization:
        im = ax.imshow(
            heatmap_log, cmap=self.annotator.colors, alpha=0.6, vmin=0, vmax=1
        )
        # im = ax.imshow(
        #     heatmap,
        #     cmap=self.annotator.colors,
        #     alpha=0.6,
        #     vmin=0,
        #     vmax=heatmap.max() * 0.8,
        # )
        ax.set_title(
            # \n({self.annotator.duration_minutes:.1f} min video, {len(valid_annotations)} samples)
            f"Bear Location Heatmap",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(
            f"{self.annotator.output_prefix}_heatmap_overlay.png",
            dpi=300,
            bbox_inches="tight",
        )

        # 2. Heatmap Only
        fig2 = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        # im = ax.imshow(heatmap, cmap=self.annotator.colors, interpolation="bilinear")
        im = ax.imshow(
            heatmap_log, cmap=self.annotator.colors, interpolation="bilinear"
        )
        ax.set_title("Space Utilization Pattern", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(
            f"{self.annotator.output_prefix}_heatmap_only.png",
            dpi=300,
            bbox_inches="tight",
        )

        # 3. Movement Path
        fig3 = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.imshow(bg_frame_rgb, alpha=0.5)

        # CHANGED: Break path at blackouts
        path_segments = []
        current_segment = []

        for ann in self.annotator.annotations:
            if ann.x == -1:  # Blackout marker
                if current_segment:
                    path_segments.append(current_segment)
                    current_segment = []
            elif ann.x and ann.x > 0:  # Valid position
                current_segment.append(ann)

        if current_segment:
            path_segments.append(current_segment)

        # Plot each segment separately
        for seg_idx, segment in enumerate(path_segments):
            if len(segment) < 2:
                continue

            x_coords = [a.x for a in segment]
            y_coords = [a.y for a in segment]

            # Use gradient colors within each segment
            colors = plt.cm.plasma(np.linspace(0, 1, len(x_coords) - 1))

            for i in range(len(x_coords) - 1):
                thickness = 1 + (i / len(x_coords)) * 3
                ax.plot(
                    x_coords[i : i + 2],
                    y_coords[i : i + 2],
                    color=colors[i],
                    linewidth=thickness,
                    alpha=0.8,
                )

            # Optional: Mark segment breaks with a small marker
            if seg_idx > 0:  # After first segment
                ax.scatter(
                    x_coords[0],
                    y_coords[0],
                    color="white",
                    s=20,
                    marker="o",
                    edgecolors="black",
                    linewidth=1,
                    zorder=5,
                )

        ax.set_title("Movement Path", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{self.annotator.output_prefix}_movement_path.png",
            dpi=300,
            bbox_inches="tight",
        )

        # 4. Contour Map
        fig4 = plt.figure(figsize=(10, 6))  # Wider for legend
        ax = plt.gca()
        ax.imshow(bg_frame_rgb, alpha=0.3)

        if heatmap[heatmap > 0].size > 0:
            levels = np.percentile(heatmap[heatmap > 0], [10, 25, 50, 75, 90])
            contour = ax.contour(
                heatmap,
                levels=levels,
                colors=["blue", "cyan", "yellow", "orange", "red"],
                linewidths=[1, 1.5, 2, 2.5, 3],
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.2f")

            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="blue", lw=2, label="10th percentile"),
                Line2D([0], [0], color="cyan", lw=2, label="25th percentile"),
                Line2D([0], [0], color="yellow", lw=2, label="50th percentile"),
                Line2D([0], [0], color="orange", lw=2, label="75th percentile"),
                Line2D([0], [0], color="red", lw=2, label="90th percentile"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                title="Activity Levels",
                framealpha=0.9,
            )

        ax.set_title("Activity Zones", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{self.annotator.output_prefix}_contour_map.png",
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"✅ Saved 4 individual plots with prefix: {self.annotator.output_prefix}"
        )
        plt.show()  # Show the last plot for quick preview
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)


class AnnotationLoader:
    """Load and plot existing annotations"""

    @staticmethod
    def load_and_plot(json_file: str):
        """Load annotations from JSON and regenerate plots"""
        print(f"Loading annotations from {json_file}...")

        with open(json_file, "r") as f:
            data = json.load(f)

        # Create mock annotator with loaded data
        video_path = data["video_path"]
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found")
            video_path = None

        # Recreate annotator object
        mock_annotator = type(
            "MockAnnotator",
            (),
            {
                "video_path": video_path,
                "width": data["video_properties"]["width"],
                "height": data["video_properties"]["height"],
                "fps": data["video_properties"]["fps"],
                "total_frames": data["video_properties"]["total_frames"],
                "duration_minutes": data["video_properties"]["duration_minutes"],
                "skip_seconds": data["annotation_settings"]["skip_seconds"],
                "output_prefix": json_file.replace("_data.json", ""),
                "annotations": [Annotation(**ann) for ann in data["annotations"]],
                "colors": LinearSegmentedColormap.from_list(
                    "fox_heatmap",
                    [
                        "#000033",
                        "#000055",
                        "#0000FF",
                        "#0066FF",
                        "#00CCFF",
                        "#66FFCC",
                        "#FFFF66",
                        "#FFCC00",
                        "#FF6600",
                        "#FF0000",
                        "#CC0000",
                    ],
                    N=100,
                ),
                "cap": cv2.VideoCapture(video_path) if video_path else None,
            },
        )()

        # Generate visualizations
        visualizer = HeatmapVisualizer(mock_annotator)
        visualizer.create_heatmap()

        print("✅ Plots regenerated successfully")


def interactive_setup():
    """Interactive setup for annotation parameters"""
    print("\n" + "=" * 50)
    print("FOX ENCLOSURE SPACE UTILIZATION TRACKER")
    print("=" * 50)

    # Mode selection
    print("\nSelect mode:")
    print("1. Annotate new video")
    print("2. Load and plot existing annotations")
    print("3. Continue previous annotation session")

    mode = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"

    if mode == "2":
        # Load existing annotations
        json_file = input("Enter JSON file path: ").strip()
        if os.path.exists(json_file):
            AnnotationLoader.load_and_plot(json_file)
            return None
        else:
            print(f"File {json_file} not found")
            return None

    elif mode == "3":
        # Continue annotation
        json_file = input("Enter JSON file to continue from: ").strip()
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
            video_path = data["video_path"]
            skip_seconds = data["annotation_settings"]["skip_seconds"]
            output_prefix = json_file.replace("_data.json", "") + "_continued"

            annotator = VideoAnnotator(video_path, skip_seconds, output_prefix)
            annotator.annotations = [Annotation(**ann) for ann in data["annotations"]]
            if annotator.annotations:
                last = annotator.annotations[-1]
                annotator.current_frame = last.frame_index + annotator.skip_frames
                annotator.last_position = (last.x, last.y) if last.x else None

            print(f"Continuing from frame {annotator.current_frame}")
            return annotator
        else:
            print(f"File {json_file} not found")
            return None

    # New annotation
    video_path = input("\nEnter video file path: ").strip()
    if not os.path.exists(video_path):
        print(f"Video file {video_path} not found")
        return None

    skip_input = input(
        f"Seconds to skip between frames [default: {DEFAULT_SKIP_SECONDS}]: "
    ).strip()
    skip_seconds = float(skip_input) if skip_input else DEFAULT_SKIP_SECONDS

    output_prefix = (
        input("Output file prefix [default: fox_tracking]: ").strip() or "fox_tracking"
    )

    print(f"\n✅ Setup complete!")
    print(f"   Video: {video_path}")
    print(f"   Skip: {skip_seconds} seconds")
    print(f"   Output: {output_prefix}_*")
    print("\nStarting annotation interface...\n")

    return VideoAnnotator(video_path, skip_seconds, output_prefix)


def main():
    """Main entry point"""
    annotator = interactive_setup()

    if annotator:
        try:
            annotator.run_annotation()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            annotator.finish()
        except Exception as e:
            print(f"\nError: {e}")
            if annotator:
                annotator.finish()


if __name__ == "__main__":
    main()