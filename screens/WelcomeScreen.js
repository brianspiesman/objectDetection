import React from 'react';
import { Button, Image, StyleSheet, Text, View } from 'react-native'

import Colors from '../constants/Colors';
import icon from "../assets/images/icon.png"

const WelcomeScreen = props => {
    return (
        <View style={styles.container}>
            <View style={styles.container}>
                <Image source={icon} style={styles.icon} />
                <Text>This is BeeMachine</Text>
                <Button 
                    title="Let's go"
                    color={Colors.primary}
                    onPress={() => props.navigation.navigate("NewObsScreen")}
                />
            </View>
            <View style={styles.container}>
                <Button
                    title="My observations"
                    color={Colors.primary}
                    onPress={() => props.navigation.navigate("ObsListScreen")}
                />
            </View>
        </View>

    );
}
export const screenOptions = navData => {
    return {
        headerTitle: "",
        headerLeft: () => 
        <View style={styles.headerButtonContainer}>
            <Button 
                title=" + "
                color='darkorange'
                onPress={() => {
                    navData.navigation.navigate({});
                }}
            />
        </View>
    }
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    icon: {
        width: 300,
        height: 300
    },
    headerButtonContainer: {
        flexDirection: 'row',
        paddingHorizontal: 5,
    }
});

export default WelcomeScreen;