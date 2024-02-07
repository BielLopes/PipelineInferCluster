const fs = require('fs');
const { InferenceSession, Tensor } = require('onnxjs');
const KNN = require('ml-knn');

async function pipeline() {
  // Load the PCA ONNX model
  const modelPathPCA = 'pca.onnx';
  const modelDataPCA = fs.readFileSync(modelPathPCA);
  const arrayBufferPCA = new Uint8Array(modelDataPCA).buffer;

  // Create an ONNX inference session
  const sessionPCA = new InferenceSession({ backendHint: 'cpu' }); // You can specify 'webgl' for GPU acceleration if supported

  // Load the PCA ONNX model into the session
  console.log(arrayBufferPCA);
  await sessionPCA.loadModel(arrayBufferPCA);
  
  // Example input data for inference
  const inputDataPCA = new Float32Array([
    816.785, 436.1925, 233.093333, 135.87, 376.393333, 284.8775, 14.8505, 8.6325, 21.872, 12.2475, 78.5625, 66.7025, 36.301333, 32.95, 65.110833,
    42.295, 27.223333, 18.66, 5.876667, 4.3575, 6.891333, 4.9275, 0.0, 0.0, 3.742333, 3.15, 94.395, 83.1725, 1.79, 2.380, 0.0, 0.0, 6.557, 5.118,
    61584.696333, 43402.595250, 158.175, 0.0, 586.028333, 381.85, 431.347667, 203.0885, 163.032333, 162.15775, 1339.155833, 871.6755, 118.12, 37.1
  ]);

  // Create an ONNX tensor from the input data
  const inputTensor = new Tensor(inputDataPCA, 'float32', [1, 48]);

  // Run inference
  const outputMapPCA = await sessionPCA.run([inputTensor]);

  // Get the output tensor
  const outputTensorPCA = outputMapPCA.values().next().value;

  // Convert the output tensor to a JavaScript array
  const outputDataPCA = outputTensorPCA.data;

  // Print the result
  console.log('Input Data:', inputDataPCA);
  console.log('Output Data:', outputDataPCA);

  // Load the KNN data training
  const featuresPath = 'features.json';
  const labelsPath = 'labels.json';

  // Load the features and labels data
  features_data = fs.readFileSync(featuresPath);
  features_data = JSON.parse(features_data);
  const pca1Data = features_data['PCA1'];
  const pca2Data = features_data['PCA2'];

  labels_data = fs.readFileSync(labelsPath);
  labels_data = JSON.parse(labels_data);
  labels_data = labels_data['labels'];

  features_dataset = [];
  label_dataset = [];

  // Iterate over the keys of each PCA object and push them as a pair into the train_dataset array
  Object.keys(pca1Data).forEach(key => {
      features_dataset.push([pca1Data[key], pca2Data[key]]);
  });
  console.log('Train dataset:', features_dataset);

  Object.keys(labels_data).forEach(key => {
      label_dataset.push(labels_data[key]);
  });
  console.log('Train dataset:', label_dataset);

  knn = new KNN(features_dataset, label_dataset, { k: 5 }); // consider 5 nearest neighbors
  ans = knn.predict([outputDataPCA[0], outputDataPCA[1]]);
  console.log(ans);
}

// Load the ONNX model and perform inference
pipeline();