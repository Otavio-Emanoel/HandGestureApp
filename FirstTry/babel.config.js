module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'], // <--- ESSA LINHA É CRUCIAL, ELA QUE ESTAVA DANDO ERRO
    plugins: [
      '@babel/plugin-transform-nullish-coalescing-operator',
      'react-native-worklets-core/plugin',
    ],
  };
};