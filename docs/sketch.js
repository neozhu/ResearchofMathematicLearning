// Daniel Shiffman
// http://codingtra.in
// http://patreon.com/codingtrain

// Polynomial Regression with TensorFlow.js
// Video: https://youtu.be/tIXDik5SGsI

let x_vals = [];
let y_vals = [];

let a, b, c, d;
let dragging = false;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);
let input;
let div;
function setup() {
  createCanvas(400, 400);
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  c = tf.variable(tf.scalar(random(-1, 1)));
  d = tf.variable(tf.scalar(random(-1, 1)));

  input = createInput();
  input.position(8, 410);
  input.style('width', '396px');
  input.value('y = ax^2 + bx + c');
  stroke(120);
  div=createDiv('<a href="https://baike.baidu.com/item/%E4%BA%8C%E6%AC%A1%E5%87%BD%E6%95%B0">二次函数（以下称函数）y = ax^2 + bx + c</a> <br> \
                ⑴参数a决定抛物线口 <br>\
                ⑵参数b决定y轴切线斜率；<br> \
                ⑶参数c决定顶点的位置：<br>\
                \
                \
                ');
  div.position(8,440);
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^2 + bx + c
  const ys = xs.square().mul(a)
     .add(xs.mul(b))
     .add(c);
  // y = ax^3 + bx^2 + cx + d
  //   const ys = xs.pow(tf.scalar(3)).mul(a)
  //     .add(xs.square().mul(b))
  //     .add(xs.mul(c))
  //     .add(d);

  return ys;
}


function mousePressed() {
  dragging = true;
}

function mouseReleased() {
  dragging = false;
}

function draw() {

    
  if (dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  } else {
    tf.tidy(() => {
      if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
        input.value(`y =  ${a.dataSync()[0].toFixed(2)}x^2 + ${b.dataSync()[0].toFixed(2)}x + ${c.dataSync()[0].toFixed(2)}`); 
        //input.value(`y = ${a.dataSync()[0].toFixed(2)}x^3 + ${b.dataSync()[0].toFixed(2)}x^2 + ${c.dataSync()[0].toFixed(2)}x + ${d.dataSync()[0].toFixed(2)}`); 
      }
    });
  }

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }


  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();
  stroke(153);
  line(0,200, 400, 200);
  line(200,0, 200, 400);

  // console.log(tf.memory().numTensors);
}