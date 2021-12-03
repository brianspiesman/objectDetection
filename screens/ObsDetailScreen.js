import React, { useState } from 'react';
import { Button, Dimensions, ScrollView, Image, View, Text, StyleSheet, TouchableWithoutFeedback, Modal } from 'react-native';
import { useSelector } from 'react-redux';
//import ReactNativeZoomableView from '@dudigital/react-native-zoomable-view/src/ReactNativeZoomableView';

//import MapPreview from '../components/MapPreview';
import Colors from '../constants/Colors';

const ObsDetailScreen = props => {
  const obsId = props.route.params.obsId;
  const selectedObs = useSelector(state =>
    state.obs.obs.find(obs => obs.id === obsId)
  );
  const [modalVisible, setModalVisible] = useState(false);

  //const selectedLocation = { lat: selectedObs.lat, lng: selectedObs.lng };

  //const showMapHandler = () => {
  //  props.navigation.navigate('MapScreen', {
  //    readonly: true,
  //    initialLocation: selectedLocation
  //  });
  //};

  return (
    <ScrollView contentContainerStyle={{ alignItems: 'center' }}>
      <TouchableWithoutFeedback onPress={() => setModalVisible(true)}>
        <View style={styles.imageContainer}>
          <Image source={{ uri: selectedObs.imageUri }} style={styles.image} />
        </View>
      </TouchableWithoutFeedback>
      <Modal visible={modalVisible} animationType='slide'>
        <Button title="Close" color={Colors.primary} onPress={() => setModalVisible(false)} />
        <View style={styles.modalContainer}>
            <Image source={{ uri: selectedObs.imageUri }} style={styles.modalImage} />
        </View>
      </Modal>
      <View style={styles.addressContainer}>
        <Text style={styles.dateContainer}>Date: {selectedObs.date}</Text>
        <Text style={styles.notesContainer}>Notes: {selectedObs.notes}</Text>
      </View>
    </ScrollView>
  );
};


//this goes in return after image
//<View style={styles.locationContainer}>
//<View style={styles.addressContainer}>
//  <Text style={styles.address}>{selectedObs.address}</Text>
//</View>
//<MapPreview
//  style={styles.mapPreview}
//  location={selectedLocation}
//  onPress={showMapHandler}
///>
//</View>

export const screenOptions = navData => {
  return {
    headerTitle: navData.route.params.obsTitle
  };
};
const deviceWidth = Dimensions.get('window').width;

const styles = StyleSheet.create({
  modalContainer: {
    flexDirection: 'row',
    backgroundColor: 'black',
    //width: '100%',
    flex:1,
  },
  modalImage: {
    width: deviceWidth,
    flex: 1,
    resizeMode: 'contain'
  },
  imageContainer: {
    flexDirection: 'row',
    backgroundColor: 'black',
    width: '100%',
  },
  image: {
    height: 300,
    flex: 1,
    resizeMode: 'contain'
  },
  notesContainer: {
    padding: 20
  },
  dateContainer: {
    padding: 20
  },
  locationContainer: {
    marginVertical: 20,
    width: '90%',
    maxWidth: 350,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: 'black',
    shadowOpacity: 0.26,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    elevation: 5,
    backgroundColor: 'white',
    borderRadius: 10
  },
  addressContainer: {
    padding: 20
  },
  address: {
    color: Colors.primary,
    textAlign: 'center'
  },
  mapPreview: {
    width: '100%',
    maxWidth: 350,
    height: 300,
    borderBottomLeftRadius: 10,
    borderBottomRightRadius: 10
  }
});

export default ObsDetailScreen;
