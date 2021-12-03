import { Platform } from 'react-native';
import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';

import WelcomeScreen, { screenOptions as welcomeScreenOptions } from '../screens/WelcomeScreen';
import ObsListScreen, { screenOptions as obsListScreenOptions } from '../screens/ObsListScreen';
import ObsDetailScreen, { screenOptions as obsDetailScreenOptions } from '../screens/ObsDetailScreen';
import NewObsScreen, { screenOptions as newObsScreenOptions } from '../screens/NewObsScreen';
//import MapScreen, { screenOptions as mapScreenOptions } from '../screens/MapScreen';
import Colors from '../constants/Colors';

const defaultNavOptions = {
    headerStyle: {
        backgroundColor: Platform.OS === 'android' ? Colors.primary : ''
    },
    headerTintColor: Platform.OS ==='android' ? 'white' : Colors.primary
};

const Stack = createStackNavigator();

const ObsNavigator = () => {
    return (
        <Stack.Navigator screenOptions={defaultNavOptions} >
            <Stack.Screen name="WelcomeScreen" component={WelcomeScreen} options={welcomeScreenOptions} />
            <Stack.Screen name="ObsListScreen" component={ObsListScreen} options={obsListScreenOptions} />
            <Stack.Screen name="ObsDetailScreen" component={ObsDetailScreen} options={obsDetailScreenOptions} />
            <Stack.Screen name="NewObsScreen" component={NewObsScreen} options={newObsScreenOptions} />

        </Stack.Navigator>
    );
};
//this goes in return after NewObsScreen
//<Stack.Screen name="MapScreen" component={MapScreen} options={mapScreenOptions} />

export default ObsNavigator;
