const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Adiciona .tflite na lista de arquivos que devem ser copiados (assets)
config.resolver.assetExts.push('tflite');

module.exports = config;