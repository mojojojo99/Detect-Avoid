function Brake() {
  $('#drone-hud video source').attr('src', "/assets/videos/Brake.mp4");
  $("#drone-hud video")[0].load();
  $("#drone-hud video").removeAttr('loop');
}

function Fly() {
  $('#drone-hud video source').attr('src', "/assets/videos/Fly.mp4");
  $("#drone-hud video")[0].load();
  $("#drone-hud video").attr('loop', "");
}

var sensor;

function requestData() {
  $.ajax({
    url: 'raspberrypi.local:5000/api/sensor',
    success: function (point) {
      var series = sensor.series[0],
        shift = series.data.length > 20;
      chart.series[0].addPoint(point, true, shift);
      setTimeout(requestData, 500);
    },
    cache: false
  });
}

$(document).ready(function () {

  sensor = Highcharts.chart('sensorhistory', {
    chart: {
      type: 'line'
    },
    plotOptions: {
      spline: {
        color: '#888888'
      }
    },
    title: {
      text: 'Obstacle Detection Distance'
    },
    series: [{
      data: []
    }]
  });

  requestData();

});