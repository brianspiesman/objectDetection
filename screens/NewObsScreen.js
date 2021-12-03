import React, { useCallback, useEffect, useState } from 'react';
import { Button, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import {AutoGrowingTextInput} from 'react-native-autogrow-textinput';
import { useDispatch } from 'react-redux';

import Colors from '../constants/Colors';
import * as obsActions from '../store/obs-actions';
import ImagePicker from '../components/ImagePicker';
//import LocationPicker from '../components/LocationPicker';

const NewObsScreen = props => {
  const [titleValue, setTitleValue] = useState('');
  const [selectedImage, setSelectedImage] = useState();

  var months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
  var day = new Date().getDate(); //Current Date
  var month = months[new Date().getMonth()]; //Current Month
  var year = new Date().getFullYear(); //Current Year
  var currentDate = month+' '+ day +', '+ year
  const [dateValue, setDateValue] = useState(currentDate);

  //const [selectedLocation, setSelectedLocation] = useState();
  const [notesValue, setNotesValue] = useState();

  

  const dispatch = useDispatch();

  const titleChangeHandler = text => { //add text validation
    setTitleValue(text);
  };

  const imageTakenHandler = imagePath => {
      setSelectedImage(imagePath);
  };

  //const locationPickedHandler = useCallback(location => {
  //  setSelectedLocation(location);
  //}, []);

  const dateChangeHandler = text => {  //add text validation
    setDateValue(text);
  };

  const notesChangeHandler = text => {
    setNotesValue(text);    //add text validation
  };

  const saveObsHandler = () => {
    dispatch(obsActions.addObs(titleValue, selectedImage, dateValue, notesValue));
    //dispatch(obsActions.addObs(titleValue, selectedImage, selectedLocation));
    props.navigation.navigate("ObsListScreen");
  };
    return (
    <ScrollView>
      <View style={styles.form}>
        <TextInput
          style={styles.textInput}
          onChangeText={titleChangeHandler}
          placeholder={'Add an observation title'}
          value={titleValue}
        />
        <ImagePicker onImageTaken={imageTakenHandler} />
        <View>
          <Text>Observation date</Text>
          <TextInput
            style={styles.textInput}
            onChangeText={dateChangeHandler}
            placeholder={'MMM DD, YYYY'}
            value={dateValue}
          />
        </View>
        <AutoGrowingTextInput 
          style={styles.textNotesInput}
          placeholder={'Add Notes'}
          maxHeight={200}
          minHeight={45}
          enableScrollToCaret
          onChangeText={notesChangeHandler}
          value={notesValue}
        />
        <TouchableOpacity style={styles.clearNotesButton} onPress={notesChangeHandler}>
            <Text>Clear</Text>
        </TouchableOpacity>
        <Button
          title="Save Observation"
          color={Colors.primary}
          onPress={saveObsHandler}
        />
      </View>
    </ScrollView>
  );
};
//this goes in return after imagepicker
//<LocationPicker navigation={props.navigation} onLocationPicked={locationPickedHandler}/>

export const screenOptions = {
  headerTitle: 'New Observation'
};

const styles = StyleSheet.create({
  form: {
    margin: 30
  },
  label: {
    fontSize: 18,
    marginBottom: 15
  },
  textInput: {
    borderBottomColor: '#ccc',
    borderBottomWidth: 1,
    marginBottom: 15,
    paddingVertical: 4,
    paddingHorizontal: 2
  },
  textNotesInput: {
    borderColor: '#ccc',
    borderWidth: 1,
    marginTop: 10,
    marginBottom: 5,
    paddingVertical: 8,
    paddingHorizontal: 8
  },
  dateInput:{
    borderColor: '#ccc',
    borderWidth: 1,
    marginTop: 10,
    marginBottom: 5,
    paddingVertical: 8,
    paddingHorizontal: 8
  },
  clearNotesButton: {
    paddingTop: 5,
    paddingBottom: 20,
    paddingHorizontal: 10,
    alignItems: 'flex-end',
    justifyContent: 'center',
  }
});

export default NewObsScreen;
