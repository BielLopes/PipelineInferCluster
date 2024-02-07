const fs = require('fs');
const { InferenceSession, Tensor } = require('onnxjs');
const KNN = require('ml-knn');
const { ScrapToElement } = require('./utils');

async function loadModel(modelPath) {
  // Load the ONNX model
  const modelData = fs.readFileSync(modelPath);
  const arrayBuffer = new Uint8Array(modelData).buffer;

  // Create an ONNX inference session
  const session = new InferenceSession({ backendHint: 'cpu' }); // You can specify 'webgl' for GPU acceleration if supported

  // Load the ONNX model into the session
  await session.loadModel(arrayBuffer);

  return session;
}

function trainKNN() {
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

  Object.keys(labels_data).forEach(key => {
      label_dataset.push(labels_data[key]);
  });

  knn = new KNN(features_dataset, label_dataset, { k: 5 }); // consider 5 nearest neighbors

  return knn;
}

async function pipeline(pca_model_path, knn_model, run_df) {
  // Load the PCA ONNX model
  const sessionPCA = await loadModel(pca_model_path);
  
  elements_df = ScrapToElement(run_df)[0];
  console.log(elements_df)
  // Example input data for inference
  const inputDataPCA = new Float32Array([
    elements_df['C_Carreg_1'], elements_df['C_Carreg_2'], elements_df['Si_Carreg_1'], elements_df['Si_Carreg_2'], elements_df['Mn_Carreg_1'], elements_df['Mn_Carreg_2'],
    elements_df['S_Carreg_1'], elements_df['S_Carreg_2'], elements_df['P_Carreg_1'], elements_df['P_Carreg_2'], elements_df['Cu_Carreg_1'], elements_df['Cu_Carreg_2'],
    elements_df['Ni_Carreg_1'], elements_df['Ni_Carreg_2'], elements_df['Cr_Carreg_1'], elements_df['Cr_Carreg_2'], elements_df['Sn_Carreg_1'], elements_df['Sn_Carreg_2'],
    elements_df['Nb_Carreg_1'], elements_df['Nb_Carreg_2'], elements_df['Mo_Carreg_1'], elements_df['Mo_Carreg_2'], elements_df['V_Carreg_1'], elements_df['V_Carreg_2'],
    elements_df['Al_Carreg_1'], elements_df['Al_Carreg_2'], elements_df['Zn_Carreg_1'], elements_df['Zn_Carreg_2'], elements_df['Pb_Carreg_1'], elements_df['Pb_Carreg_2'],
    elements_df['Hg_Carreg_1'], elements_df['Hg_Carreg_2'], elements_df['N_Carreg_1'], elements_df['N_Carreg_2'], elements_df['Fe_Carreg_1'], elements_df['Fe_Carreg_2'],
    elements_df['FeO_Carreg_1'], elements_df['FeO_Carreg_2'], elements_df['Rust_Carreg_1'], elements_df['Rust_Carreg_2'], elements_df['OilGreasesRubber_Carreg_1'],
    elements_df['OilGreasesRubber_Carreg_2'], elements_df['PaintingsCoatings_Carreg_1'], elements_df['PaintingsCoatings_Carreg_2'], elements_df['NonMetalics_Carreg_1'],
    elements_df['NonMetalics_Carreg_2'], elements_df['H2O_Carreg_1'], elements_df['H2O_Carreg_2']
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

  // Perform inference using the KNN model
  ans = knn_model.predict([outputDataPCA[0], outputDataPCA[1]]);
  console.log(ans);
}

// Load the ONNX model and perform inference
pca_model_path = 'pca.onnx';
knn_model = trainKNN();
run_df = [
  {
    'BatchCode': 41397834,
    'SARP_Carreg_1': 0,
    'SARP_Carreg_2': 2150,
    'SCAVA_Carreg_1': 2050,
    'SCAVA_Carreg_2': 0,
    'SEMR_Carreg_1': 0,
    'SEMR_Carreg_2': 0,
    'SEST_Carreg_1': 5600,
    'SEST_Carreg_2': 0,
    'SGUS_Carreg_1': 15600,
    'SGUS_Carreg_2': 6100,
    'SHD_Carreg_1': 26450,
    'SHD_Carreg_2': 26400,
    'SMST_Carreg_1': 2100,
    'SMST_Carreg_2': 2900,
    'SPES_Carreg_1': 3150,
    'SPES_Carreg_2': 0,
    'SREC_Carreg_1': 4950,
    'SREC_Carreg_2': 0,
    'SSHR_Carreg_1': 0,
    'SSHR_Carreg_2': 0,
    'STES_Carreg_1': 6000,
    'STES_Carreg_2': 3600,
    'SUCG_Carreg_1': 0,
    'SUCG_Carreg_2': 0
  }];

pipeline(pca_model_path, knn_model, run_df);