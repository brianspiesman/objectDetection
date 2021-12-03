import * as SQLite from 'expo-sqlite';

const db = SQLite.openDatabase('obs.db');

export const init = () => {
    const promise = new Promise((resolve, reject) => {
        db.transaction((tx) => {
            tx.executeSql(
                //'CREATE TABLE IF NOT EXISTS obs (id INTEGER PRIMARY KEY NOT NULL, title TEXT NOT NULL, imageUri TEXT NOT NULL, address TEXT NOT NULL, lat REAL NOT NULL, lng REAL NOT NULL);',
                'CREATE TABLE IF NOT EXISTS obs (id INTEGER PRIMARY KEY NOT NULL, title TEXT, imageUri TEXT NOT NULL, date TEXT, notes TEXT);',
                [],
                () => {
                    resolve();
                },
                (_, err) => {
                    reject(err);
                },
    
            );
        });
    });
    return promise;
};

export const insertObs = (title, imageUri, date, notes) => {
//export const insertObs = (title, imageUri, address, lat, lng) => {
    const promise = new Promise((resolve, reject) => {
        db.transaction((tx) => {
            tx.executeSql(
                `INSERT INTO obs (title, imageUri, date, notes) VALUES (?, ?, ?, ?);`,
                //`INSERT INTO obs (title, imageUri, address, lat, lng) VALUES (?, ?, ?, ?, ?);`,
                [title, imageUri, date,notes],
                //[title, imageUri, address, lat, lng],
                (_, result) => {
                    resolve(result);
                },
                (_, err) => {
                    reject(err);
                },
            );
        });
    });
    return promise;
};

export const fetchObs = () => {
    const promise = new Promise((resolve, reject) => {
        db.transaction((tx) => {
            tx.executeSql(
                'SELECT * FROM obs',
                [],
                (_, result) => {
                    resolve(result);
                },
                (_, err) => {
                    reject(err);
                },
            );
        });
    });
    return promise;
}