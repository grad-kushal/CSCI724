function initMap() {
    var map = new google.maps.Map(document.getElementById("map"), {
      zoom: 8,
      center: { lat: 37.7749, lng: -122.4194 }
    });

    var marker;
    google.maps.event.addListener(map, "click", function(event) {
      marker = new google.maps.Marker({
        position: event.latLng,
        map: map
      });
    });
  }
