import React, { useEffect } from 'react';
import { Button, FlatList, StyleSheet, Text, View } from 'react-native';
import { Colors } from 'react-native/Libraries/NewAppScreen';
import { useSelector, useDispatch } from 'react-redux'

import ObsItem from '../components/ObsItem';
import * as obsActions from '../store/obs-actions';

const ObsListScreen = props => {
    const obs = useSelector(state => state.obs.obs);
    const dispatch = useDispatch();

    useEffect(() => {
        dispatch(obsActions.loadObs());
    }, [dispatch]);

    return (
        <FlatList
            data={obs}
            keyExtractor={item => item.id}
            renderItem={itemData => (
                <ObsItem
                    image={itemData.item.imageUri}
                    title={itemData.item.title}
                    date={itemData.item.date}
                    //address={itemData.item.address}
                    onSelect={() => {
                        props.navigation.navigate('ObsDetailScreen', {
                            obsTitle: itemData.item.title,
                            obsId: itemData.item.id
                        });
                    }} 
                />
            )}
       />
    )
}

export const screenOptions = navData => {
    return {
        headerTitle: "",
        headerRight: () => 
        <View style={styles.headerButtonContainer}>
            <View style={styles.headerButton}>
                <Button 
                    style={styles.headerButton}
                    title=" + "
                    color='darkorange'
                    onPress={() => {
                        navData.navigation.push('NewObsScreen');
                    }}
                />
            </View>
            <View style={styles.headerButton}>
                <Button
                    color="black"
                    title="Home"
                    onPress={() => {
                        navData.navigation.popToTop();
                    }}
                />
            </View>
        </View>
    };
};

const styles = StyleSheet.create({
    headerButtonContainer: {
        flexDirection: 'row'
    },
    headerButton: {
        paddingHorizontal: 5,
    },
});

export default ObsListScreen;
