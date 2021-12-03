class Obs {
    //constructor(id, title, imageUri, address, lat, lng) {
    constructor(id, title, imageUri, date, notes) {
        this.id = id;
        this.title = title;
        this.imageUri = imageUri;
        this.date = date;
        this.notes = notes;
        //this.address = address;
        //this.lat = lat;
        //this.lng = lng;
    }
}

export default Obs;
