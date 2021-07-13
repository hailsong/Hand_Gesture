var Links = {
  setColor:function(color){
      // var alist = document.querySelectorAll('a');
      // var i = 0;
      // while(i < alist.length){
      //   alist[i].style.color = color;
      //   i = i + 1;
      // }
      $('a').css('color', color);
    }
  }

  var Body = {
    setColor:function (color){
      // document.querySelector('body').style.backgroundColor = color;
      $('body').css('color', color)
    },
    setGroundColor:function (color){
      // document.querySelector('body').style.color = color;
      $('body').css('backgroundColor', color)
    }
  }
  function daynightHandle(self){
    var target = document.querySelector('body');
    if(self.value === 'night'){
      Body.setColor('black');
      Body.setGroundColor('white');
      self.value = 'day';

      Links.setColor('blue')
      }
     else {
      Body.setColor('white');
      Body.setGroundColor('black');
      self.value = 'night';

      Links.setColor('powderblue')
      }
    }
