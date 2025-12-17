import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, ScrollView } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useEffect, useCallback, useRef, useMemo, useState } from 'react';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';

const HAND_MODEL = require('./assets/hand_landmark_full.tflite');
const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [5, 9], [9, 10], [10, 11], [11, 12], // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17], // Palm base to pinky base
];

export default function App() {
  const device = useCameraDevice('front');
  const { hasPermission, requestPermission } = useCameraPermission();
  const { resize } = useResizePlugin();

  const [handStatus, setHandStatus] = useState<string>('Aguardando detecção...');
  const [landmarks, setLandmarks] = useState<Array<{x:number;y:number;z:number}>>([]);
  const debugCountRef = useRef(0);

  const [previewSize, setPreviewSize] = useState({ width: 170, height: 220 });

  const model = useTensorflowModel(HAND_MODEL);

  const lastLogRef = useRef<number>(0);

  const onFrame = useMemo(
    () =>
      Worklets.createRunOnJS((timestamp: number) => {
        // menos ruído: loga raramente
        const now = Date.now();
        if (now - lastLogRef.current > 4000) {
          lastLogRef.current = now;
          console.log('Frame tick');
        }
      }),
    [],
  );

  const onLandmarks = useMemo(
    () =>
      Worklets.createRunOnJS((flat: number[]) => {
        if (!flat?.length) return;

        const pts = [] as Array<{x:number;y:number;z:number}>;
        for (let i = 0; i + 2 < flat.length; i += 3) {
          pts.push({ x: flat[i], y: flat[i + 1], z: flat[i + 2] });
        }

        // Heurística simples para filtrar falsos positivos quando não há mão.
        const norm = pts.map((p) => ({ x: p.x / 224, y: p.y / 224, z: p.z }));
        const xs = norm.map((p) => p.x);
        const ys = norm.map((p) => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const w = maxX - minX;
        const h = maxY - minY;
        const area = w * h;
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const meanDist = norm.reduce((acc, p) => acc + Math.hypot(p.x - cx, p.y - cy), 0) / norm.length;

        const validBox = area > 0.02 && area < 0.45 && w > 0.08 && h > 0.08;
        const validSpread = meanDist > 0.05;
        const inBounds = norm.every((p) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);

        if (!validBox || !validSpread || !inBounds) {
          setLandmarks([]);
          setHandStatus('Mão não detectada');
          return;
        }

        setLandmarks(pts);
        setHandStatus('Mão detectada');
      }),
    [],
  );

  const mappedLandmarks = useMemo(() => {
    const { width, height } = previewSize;
    const isFront = device?.position === 'front';
    return landmarks.map((p) => {
      const xClamped = Math.max(0, Math.min(224, p.x));
      const yClamped = Math.max(0, Math.min(224, p.y));
      const nx = xClamped / 224;
      const ny = yClamped / 224;
      const x = (isFront ? 1 - nx : nx) * width;
      const y = ny * height;
      return { x, y, z: p.z };
    });
  }, [landmarks, previewSize, device?.position]);

  const onDebug = useMemo(
    () =>
      Worklets.createRunOnJS((msg: any) => {
        // limitar logs para não poluir
        if (debugCountRef.current >= 3) return;
        debugCountRef.current += 1;
        try {
          console.log('DEBUG(frame):', JSON.stringify(msg).slice(0, 400));
        } catch (e) {
          console.log('DEBUG(frame):', msg);
        }
      }),
    [],
  );

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';

    // Log de timestamp controlado
    onFrame(frame.timestamp);

    // Modelo ainda não carregou? pule
    if (model.state !== 'loaded') return;

    // Redimensiona frame para o esperado pelo modelo (224x224 RGB float32)
    const input = resize(frame, {
      scale: { width: 224, height: 224 },
      pixelFormat: 'rgb',
      dataType: 'float32',
    });

    const output = model.model?.runSync([input]);

    // As variantes do modelo costumam devolver um array único com 63 floats (21 landmarks * 3)
    const first = output?.[0] as any;
    const data: number[] | Float32Array | undefined = first?.data ?? first;

    if (data) {
      // Debug: sempre envie uma pequena amostra para o JS para inspecionar o formato
      try {
        onDebug({ len: data.length, sample: Array.from(data.slice(0, 12)) });
      } catch (e) {}

      if (data.length >= 63) {
        // Evita enviar muitos eventos para JS: amostra a cada ~3 frames
        const ts = Number(frame.timestamp ?? 0);
        if (Number.isFinite(ts) && ts % 3 === 0) {
          onLandmarks(Array.from(data.slice(0, 63)));
        }
      }
    } else {
      // No output: debug
      try {
        onDebug({ len: 0, note: 'no output' });
      } catch (e) {}
    }
  }, [onFrame, onLandmarks, onDebug, model.state, resize]);

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
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.hero}>
          <Text style={styles.kicker}>Controle por gestos</Text>
          <Text style={styles.title}>Navegue sem tocar na tela</Text>
          <Text style={styles.body}>
            Use gestos simples para scrollar e interagir. Mantenha a câmera apontada para sua mão e
            acompanhe o preview no canto.
          </Text>
          <View style={styles.tagRow}>
            <View style={styles.tag}><Text style={styles.tagText}>Vision Camera</Text></View>
            <View style={styles.tag}><Text style={styles.tagText}>Worklets</Text></View>
            <View style={styles.tag}><Text style={styles.tagText}>On-device</Text></View>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Como funciona?</Text>
          <Text style={styles.cardBody}>
            • Gestos reconhecidos por frame processor (offline).
          </Text>
          <Text style={styles.cardBody}>
            • Modelo de mãos (MediaPipe/TFLite) para landmarks.
          </Text>
          <Text style={styles.cardBody}>
            • Mapeie gestos para scroll, taps virtuais e navegação.
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Gestos suportados (exemplo)</Text>
          <Text style={styles.cardBody}>• Swipe para cima/baixo: scroll suave.</Text>
          <Text style={styles.cardBody}>• Mão aberta vs fechada: play/pause ou seleção.</Text>
          <Text style={styles.cardBody}>• Pinch: tap virtual ou zoom.</Text>
          <Text style={styles.cardBody}>• Mão fora do quadro: pausa automática.</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Dicas de uso</Text>
          <Text style={styles.cardBody}>• Iluminação razoável ajuda o modelo.</Text>
          <Text style={styles.cardBody}>• Mantenha a mão a ~40–70 cm da câmera.</Text>
          <Text style={styles.cardBody}>• Evite fundos muito brilhantes ou com padrão pesado.</Text>
          <Text style={styles.cardBody}>• Use gestos amplos para melhor detecção.</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Próximos passos</Text>
          <Text style={styles.cardBody}>• Integrar modelo TFLite/MediaPipe para landmarks.</Text>
          <Text style={styles.cardBody}>• Debounce e smoothing para gestos mais estáveis.</Text>
          <Text style={styles.cardBody}>• Mapear eventos para ScrollView/FlatList.</Text>
          <Text style={styles.cardBody}>• Adicionar feedback visual ao reconhecer gestos.</Text>
        </View>

        <View style={styles.statusCard}>
          <Text style={styles.statusTitle}>Status da mão</Text>
          <Text style={styles.statusBody}>{model.state === 'loaded' ? handStatus : 'Carregando modelo...'}</Text>
        </View>
      </ScrollView>

      <View style={styles.cameraContainer}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive
          frameProcessor={frameProcessor}
          pixelFormat="yuv"
        />
        <Text style={styles.previewLabel}>Preview</Text>
        {/* Landmark overlay (normalized coordinates expected: x,y in [0..1]) */}
        <View
          style={styles.landmarkOverlay}
          pointerEvents="none"
          onLayout={(e) => {
            const { width, height } = e.nativeEvent.layout;
            if (width && height) {
              setPreviewSize({ width, height });
            }
          }}
        >
          {mappedLandmarks
            .filter((_, idx) => [4, 8, 12, 16, 20].includes(idx))
            .map((p, i) => {
              const left = Math.max(0, Math.min(previewSize.width - 12, p.x - 6));
              const top = Math.max(0, Math.min(previewSize.height - 12, p.y - 6));
              return (
                <View
                  key={`tip-${i}`}
                  style={[styles.landmarkDot, { left, top }]}
                />
              );
            })}

          {mappedLandmarks.length > 0 && (
            <Text style={styles.countBadge}>5 pontas</Text>
          )}

          {mappedLandmarks.length === 0 && (
            <Text style={styles.noHandText}>Mostre a mão na câmera</Text>
          )}
        </View>
      </View>

      <StatusBar style="light" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0b1021',
    paddingHorizontal: 24,
    paddingTop: 24,
    paddingBottom: 24 + 200,
  },
  scrollContent: {
    paddingBottom: 120,
    gap: 16,
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0b1021',
  },
  hero: {
    backgroundColor: '#101936',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
  },
  kicker: {
    color: '#7dd3fc',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
  },
  title: {
    color: '#e2e8f0',
    fontSize: 26,
    fontWeight: '700',
    marginBottom: 8,
  },
  body: {
    color: '#cbd5e1',
    fontSize: 15,
    lineHeight: 22,
  },
  tagRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 12,
  },
  tag: {
    backgroundColor: 'rgba(125, 211, 252, 0.12)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  tagText: {
    color: '#7dd3fc',
    fontWeight: '600',
    fontSize: 12,
  },
  card: {
    backgroundColor: '#0f172a',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.05)',
    gap: 6,
  },
  cardTitle: {
    color: '#e2e8f0',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 4,
  },
  cardBody: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 20,
  },
  statusCard: {
    backgroundColor: '#0f172a',
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.35)',
    gap: 6,
  },
  statusTitle: {
    color: '#7dd3fc',
    fontSize: 16,
    fontWeight: '700',
  },
  statusBody: {
    color: '#e2e8f0',
    fontSize: 14,
  },
  cameraContainer: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    width: 170,
    height: 220,
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(125, 211, 252, 0.35)',
    backgroundColor: '#0b1021',
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
  },
  previewLabel: {
    position: 'absolute',
    top: 8,
    left: 10,
    backgroundColor: 'rgba(0,0,0,0.4)',
    color: '#e2e8f0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    fontSize: 12,
    fontWeight: '600',
  },
  landmarkOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  landmarkDot: {
    position: 'absolute',
    width: 12,
    height: 12,
    borderRadius: 12,
    backgroundColor: '#38bdf8',
    borderWidth: 1,
    borderColor: '#0b1021',
  },
  countBadge: {
    position: 'absolute',
    top: 8,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
    color: '#e2e8f0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    fontSize: 12,
    fontWeight: '600',
  },
  noHandText: {
    position: 'absolute',
    bottom: 8,
    left: 10,
    color: '#cbd5e1',
    fontSize: 12,
    backgroundColor: 'rgba(0,0,0,0.4)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
});
