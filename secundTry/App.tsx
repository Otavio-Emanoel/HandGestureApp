import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useEffect, useCallback, useRef, useMemo } from 'react';

export default function App() {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();

  const lastLogRef = useRef<number>(0);

  const onFrame = useMemo(
    () =>
      Worklets.createRunOnJS((timestamp: number) => {
        const now = Date.now();
        if (now - lastLogRef.current > 500) {
          lastLogRef.current = now;
          console.log('Frame timestamp (ns):', timestamp);
        }
      }),
    [],
  );

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    onFrame(frame.timestamp);
  }, [onFrame]);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text>Conceda permissão para acessar a câmera.</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text>Carregando câmera...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />
      <StatusBar style="light" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
  },
});
