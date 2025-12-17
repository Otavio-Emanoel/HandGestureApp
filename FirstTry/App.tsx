import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { 
  Camera, 
  useCameraDevice, 
  useFrameProcessor, 
} from 'react-native-vision-camera';
import { useTensorflowModel, Tensor } from 'react-native-fast-tflite';
// Corrigido: O import permanece, mas agora haverá declaração de tipos
import { useResizePlugin } from 'vision-camera-resize-plugin'; 
import { Worklets } from 'react-native-worklets-core';


// IMPORTANTE: Use o nome exato do arquivo que você extraiu
const MODEL_ASSET = require('./assets/hand_landmarks_detector.tflite');

export default function App() {
  const device = useCameraDevice('front');
  const [hasPermission, setHasPermission] = useState(false);
  
  // 1. Carrega o modelo
  const model = useTensorflowModel(MODEL_ASSET);
  
  // 2. Carrega o plugin de redimensionamento
  const { resize } = useResizePlugin(); 

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    
    if (model.state !== 'loaded') return;

    // 3. Preparar a imagem (O modelo do Google exige 224x224 float32)
    // O resize plugin converte o frame da câmera (yuv) para o formato que o modelo entende (rgb float)
    const input = resize(frame, {
      scale: {
        width: 224, 
        height: 224
      },
      pixelFormat: 'rgb',
      dataType: 'float32' // Normaliza os pixels entre 0.0 e 1.0 ou 0 e 255 dependendo do modelo
    });

    // 4. Rodar o modelo
    // O runSync executa a detecção nesse frame específico
    const output = model.model.runSync([input]);

    // 5. Ver o resultado
    // O output é um array de tensores. O output[0] geralmente contém as coordenadas.
    const landmarks = output[0];
    
    // ATENÇÃO: Imprimir no console aqui pode travar o app se for muito rápido.
    // Use com moderação ou filtre (ex: a cada 60 frames).
    console.log(`Detectou algo? ${landmarks.length > 0}`);

  }, [model]);

  if (!hasPermission) return <Text>Sem permissão</Text>;
  if (device == null) return <Text>Carregando...</Text>;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
});