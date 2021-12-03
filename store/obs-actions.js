import * as FileSystem from 'expo-file-system';

import { insertObs, fetchObs } from '../helpers/db';
//import ENV from '../env';

export const ADD_OBS = 'ADD_OBS';
export const SET_OBS = 'SET_OBS';

//export const addObs = (title, image, location) => {
export const addObs = (title, image, date, notes) => {
  return async dispatch => {
    //const response = await fetch(`https://maps.googleapis.com/maps/api/geocode/json?latlng=${location.lat},${location.lng}&key=${ENV.googleApiKey}`);
    
    //if (!response.ok) {
    //  throw new Error('Something went wrong!');
    //}

    //const resData = await response.json();
    //if (!resData.results) {
    //  throw new Error('Something went wrong!');
    //}

    //const address = resData.results[0].formatted_address;

    const fileName = image.split('/').pop();
    const newPath = FileSystem.documentDirectory + fileName;

    try {
      await FileSystem.moveAsync({
        from: image,
        to: newPath
      });
      const dbResult = await insertObs(
        title,
        newPath,
        date,
        notes,
        //address,
        //location.lat,
        //location.lng,
      );
      console.log(dbResult);
      dispatch({ 
        type: ADD_OBS, 
        obsData: { 
          id: dbResult.insertId, 
          title: title, 
          image: newPath,
          date: date,
          notes: notes, 
          //address: address, 
          //coords: {
          //  lat:location.lat,
          //  lng:location.lng
          //} 
        } 
      });
    } catch (err) {
      console.log(err);
      throw err;
    }
  };
};

export const loadObs = () => {
    return async dispatch => {
        try {
            const dbResult = await fetchObs();
            console.log(dbResult);
            dispatch({ type: SET_OBS, obs: dbResult.rows._array });
        } catch (err) {
            throw err;
        }
    };
};