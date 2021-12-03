import React from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';
import Colors from '../constants/Colors';

const ObsItem = props => {
  return (
    <TouchableOpacity onPress={props.onSelect} style={styles.obsItem}>
      <Image style={styles.image} source={{ uri: props.image }} />
      <View style={styles.infoContainer}>
        <Text style={styles.title}>{props.title}</Text>
        <Text style={styles.date}>{props.date}</Text>
        <Text style={styles.notes}>{props.notes}</Text>

      </View>
    </TouchableOpacity>
  );
};
//this goes in return after text
//<Text style={styles.address}>{props.address}</Text>

const styles = StyleSheet.create({
  obsItem: {
    borderBottomColor: '#ccc',
    borderBottomWidth: 1,
    paddingVertical: 12,
    paddingHorizontal: 25,
    flexDirection: 'row',
    alignItems: 'center'
  },
  image: {
    width: 75,
    height: 75,
    borderRadius: 7,
    backgroundColor: '#ccc',
    borderColor: Colors.primary,
    borderWidth: 1
  },
  infoContainer: {
    marginLeft: 25,
    width: 250,
    justifyContent: 'center',
    alignItems: 'flex-start'
  },
  title: {
    color: 'black',
    fontSize: 18,
    marginBottom: 3
  },
  notes: {
    color: '#666',
    fontSize: 14,
  },
  date: {
    color: '#666',
    fontSize: 14,
  },
  //address: {
  //  color: '#666',
  //  fontSize: 16
  //}
});

export default ObsItem;
