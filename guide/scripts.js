
    var idx = 0;
    var i = 0; // 사진 인덱스를 저장할 변수
    $(".pre").click(function() { // img 크기만큼 왼쪽으로 이동
      idx = idx - 1;
      if (idx < 1) {
        i = idx % 3;
        i = i + 4;
        if (i == 4) {
          i = 1;
        }
      } else {
        i = idx % 3;
        if (i == 0) {
          i = 3;
        }
      }
      $(".imgSlide>li:last-child").remove();
      $(".imgSlide").prepend("<li><img src='http://doqtqu.dothome.co.kr/images/imgSlideBtn/images(" + i + ").jpg' alt=''></li>");
      $(".imgSlide").css({
        "left": "-3200px"
      });
      $(".imgSlide").stop().animate({
        "left": "-2400px"
      }, "slow");
      console.log(idx);
    });
    $(".next").click(function() { // img 크기만큼 오른쪽으로 이동
      idx = idx + 1;
      if (idx < 1) {
        i = idx % 3;
        i = i + 3;
      } else {
        i = idx % 3;
        if (i == 0) {
          i = 3;
        }
      }
      $(".imgSlide>li:first-child").remove();
      $(".imgSlide").append("<li><img src='http://doqtqu.dothome.co.kr/images/imgSlideBtn/images(" + i + ").jpg' alt=''></li>");
      $(".imgSlide").css({
        "left": "-1600px"
      });
      $(".imgSlide").stop().animate({
        "left": "-2400px"
      }, "slow");
      console.log(idx);
    });
