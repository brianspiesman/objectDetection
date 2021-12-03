import React from 'react';
import { createStore, combineReducers, applyMiddleware } from 'redux';
import { Provider } from 'react-redux';
import ReduxThunk from 'redux-thunk';
import { NavigationContainer } from '@react-navigation/native';

import ObsNavigator from './navigation/ObsNavigator';
import obsReducer from './store/obs-reducer';
import { init } from './helpers/db';

init().then(() => {
  console.log('Initialized database');
}).catch(err => {
  console.log('Initialized db failed');
  console.log(err);
});

const rootReducer = combineReducers({
  obs: obsReducer
});

const store = createStore(rootReducer, applyMiddleware(ReduxThunk));

export default function App() {
  return (
    <NavigationContainer>
      <Provider store={store}>
        <ObsNavigator />
      </Provider>
    </NavigationContainer>

  );
}
