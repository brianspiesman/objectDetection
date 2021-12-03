import { ADD_OBS, SET_OBS } from './obs-actions';
import Obs from '../models/obs';

const initialState = {
  obs: []
};

export default (state = initialState, action) => {
  switch (action.type) {
    case SET_OBS:
      return {
        obs: action.obs.map(
          pl => new Obs(
            pl.id.toString(), 
            pl.title, 
            pl.imageUri,
            pl.date,
            pl.notes,
            //pl.address, 
            //pl.lat, 
            //pl.lng,
          )
        )
      };
    case ADD_OBS:
      const newObs = new Obs(
        action.obsData.id.toString(),
        action.obsData.title,
        action.obsData.image,
        action.obsData.date,
        action.obsData.notes,
        //action.obsData.address,
        //action.obsData.coords.lat,
        //action.obsData.coords.lng,
      );
      return {
        obs: state.obs.concat(newObs)
      };
    default:
      return state;
  }
};