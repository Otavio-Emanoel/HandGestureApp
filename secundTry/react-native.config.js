module.exports = {
  dependencies: {
    'react-native-fast-tflite': {
      platforms: {
        android: {
          // Disable CMake autolinking (library doesn't ship a CMakeLists)
          cmakeListsPath: '',
        },
      },
    },
    'vision-camera-resize-plugin': {
      platforms: {
        android: {
          // Disable CMake autolinking (no codegen CMakeLists provided)
          cmakeListsPath: '',
        },
      },
    },
    'react-native-worklets-core': {
      platforms: {
        android: {
          // Disable CMake autolinking for codegen path
          cmakeListsPath: '',
        },
      },
    },
  },
};
