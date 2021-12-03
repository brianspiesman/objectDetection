import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button, Image, StyleSheet, Text, View } from 'react-native';
import Colors from '../constants/Colors';
import * as ImagePicker from 'expo-image-picker';
import { decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system';
import * as jpeg from 'jpeg-js'


const ImgPicker = props => {
  const [pickedImage, setPickedImage] = useState();
  const [displayText, setDisplayText] = useState('Loading Vision Model')
  const [model, setModel] = useState(null)

    useEffect(() => {
        (async () => {
          await tf.ready();

          const modelJson = require("../assets/VisModels/model_OD_mobnetV2_640.json")
          const modelWeight = require("../assets/VisModels/model_OD_mobnetV2_640_weights.bin")
          const model = await tf.loadGraphModel(bundleResourceIO(modelJson,modelWeight))
          setModel(model)
          setDisplayText("Vision Model Ready!")
        })();
    }, []);

    useEffect(() => {
        (async () => {
          if (Platform.OS !== 'web') {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (status !== 'granted') {
              alert('Sorry, BeeMachine needs camera roll permissions!');
            }
          }
        })();
      }, []);

    useEffect(() => {
    (async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
            alert('Sorry, BeeMachine needs camera permissions!');
        }
    })();
    }, []);

    

    async function getPrediction(){
          const result = await ImagePicker.launchImageLibraryAsync({
          quality: 1,
          exif: true,
        });
    
        if (!result.cancelled) {
          setPickedImage(result.uri);
        }
        props.onImageTaken(result.uri);
        console.log(result)
        setDisplayText("Processing image")

        const fileUri = result.uri;      
        const IMGSIZE = 640
    
        const imgB64 = await FileSystem.readAsStringAsync(fileUri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
        const imageData = new Uint8Array(imgBuffer)
        const imageTensor = decodeJpeg(imageData).expandDims().resizeBilinear([IMGSIZE,IMGSIZE]).div(tf.scalar(255)).reshape([1,IMGSIZE,IMGSIZE,3])

        console.log(imageTensor)
        setDisplayText("Locating bee")      
    
        //const prediction = await model.predict(imageTensor).data()
        const modelExeAsync = await model.executeAsync(imageTensor);
        const prediction = modelExeAsync.dataSync();

        setDisplayText(prediction)
      }

    return <View style={styles.imagePicker}>
        <Text style={styles.dispText}>{displayText}</Text>
        <View style={styles.imagePreview}>
            {!pickedImage ? (
                <Text>No image picked yet.</Text>
            ) : (
                <Image style={styles.image} source={{uri: pickedImage}} />
            )}
        </View>
        <View style={styles.buttonContainer}>
          <Button
              title="Get Image"
              color={Colors.primary}
              onPress={getPrediction}
          />
        </View>
    </View>
};

const styles  = StyleSheet.create({
    buttonContainer:{
      justifyContent: 'space-evenly',
      flexDirection: 'row',
      width: '100%',
    },    
    imagePicker: {
        alignItems: 'center',
        marginBottom: 5,
    },
    imagePreview: {
        width: '100%',
        height: 250,
        marginBottom: 10,
        justifyContent: 'center',
        alignItems: 'center',
        borderColor: '#ccc',
        borderWidth: 1,
    },
    image:{
        width: '100%',
        height: '100%',
    },
});

export default ImgPicker;