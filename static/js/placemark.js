ymaps.ready(init);

var myMap;
function init () {
    myMap = new ymaps.Map('map', {
        center: [43.238949, 76.889709],
        zoom: 10
    }, {
        searchControlProvider: 'yandex#search'
    });
}