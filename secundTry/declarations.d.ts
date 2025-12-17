declare module 'vision-camera-resize-plugin' {
  import type { Frame } from 'react-native-vision-camera';

  type ResizeOptions = {
    scale: { width: number; height: number };
    pixelFormat?: 'rgb' | 'rgba' | 'yuv';
    dataType?: 'float32' | 'uint8';
  };

  type ResizeFn = (frame: Frame, options: ResizeOptions) => any;

  export function useResizePlugin(): { resize: ResizeFn };
}
