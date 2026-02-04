import { loadOml2d } from '../dist/index.js';

// 根据屏幕大小计算合适的缩放比例
function getOptimalScale() {
  const screenWidth = window.innerWidth;
  const screenHeight = window.innerHeight;
  const minDimension = Math.min(screenWidth, screenHeight);
  
  // 基础缩放比例，根据屏幕大小调整
  let baseScale = 0.06;
  
  if (minDimension < 768) {
    baseScale = 0.04; // 小屏幕
  } else if (minDimension < 1024) {
    baseScale = 0.05; // 中等屏幕
  } else {
    baseScale = 0.06; // 大屏幕
  }
  
  return baseScale;
}

const oml2d = loadOml2d({
  importType: 'cubism5',
  models: [
    {
      path: '/mao/mao_pro.model3.json',
      scale: getOptimalScale(), // 自适应缩放比例
      position: [0, 0], // 居中显示
      showHitAreaFrames: false, // 关闭调试框，更干净
      motionPreloadStrategy: 'ALL' // 预加载所有动作，获得更好体验
    }
  ]
});

// mao模型加载成功后的交互设置
oml2d.onStageSlideIn(() => {
  oml2d.tipsMessage('mao_pro_en模型加载成功！🐱', 3000, 10);
  
  // 3秒后播放欢迎动作
  setTimeout(() => {
    if (oml2d.models && oml2d.models.playMotion) {
      oml2d.models.playMotion(''); // 使用空字符串动作组
    }
  }, 1500);
});

// 添加点击交互
document.addEventListener('click', (e) => {
  // 随机播放mao的表情
  const expressions = ['exp_01', 'exp_02', 'exp_03', 'exp_04', 'exp_05', 'exp_06', 'exp_07', 'exp_08'];
  const randomExp = expressions[Math.floor(Math.random() * expressions.length)];
  
  // 随机播放动作
  const motions = ['mtn_01', 'mtn_02', 'mtn_03', 'mtn_04', 'special_01', 'special_02', 'special_03'];
  const randomMotion = motions[Math.floor(Math.random() * motions.length)];
  
  // 播放表情（当前版本不支持）
  console.warn('playExpression方法在当前版本中不存在');
  // 播放动作
  if (oml2d.models && oml2d.models.playMotion) {
    oml2d.models.playMotion(randomMotion);
  }
  
  oml2d.tipsMessage(`表情: ${randomExp} | 动作: ${randomMotion}`, 2000, 10);
});

// 键盘快捷键
let currentExpressionIndex = 0;
let currentMotionIndex = 0;

document.addEventListener('keydown', (e) => {
  switch(e.key) {
    case 'e': // 表情切换（当前版本不支持）
      currentExpressionIndex = (currentExpressionIndex + 1) % 8;
      const expName = `exp_0${currentExpressionIndex + 1}`;
      console.warn('playExpression方法在当前版本中不存在');
      oml2d.tipsMessage(`表情切换: ${expName}（当前版本不支持）`, 1500, 10);
      break;
      
    case 'm': // 动作切换
      const motions = ['mtn_01', 'mtn_02', 'mtn_03', 'mtn_04', 'special_01', 'special_02', 'special_03'];
      currentMotionIndex = (currentMotionIndex + 1) % motions.length;
      const motionName = motions[currentMotionIndex];
      if (oml2d.models && oml2d.models.playMotion) {
        oml2d.models.playMotion(motionName);
        oml2d.tipsMessage(`动作: ${motionName}`, 1500, 10);
      }
      break;
      
    case 'r': // 随机动作（表情当前版本不支持）
      const randomMotions = ['mtn_01', 'mtn_02', 'mtn_03', 'mtn_04', 'special_01', 'special_02', 'special_03'];
      const randomMotion = randomMotions[Math.floor(Math.random() * randomMotions.length)];
      
      console.warn('playExpression方法在当前版本中不存在');
      if (oml2d.models && oml2d.models.playMotion) {
        oml2d.models.playMotion(randomMotion);
        oml2d.tipsMessage(`随机动作: ${randomMotion}`, 2000, 10);
      }
      break;
  }
});

oml2d.onStageSlideIn(() => {
  oml2d.tipsMessage('模型加载成功！', 2000, 10);
});

// oml2d.onStageSlideIn(() => {
//   oml2d.loadNextModel();
// });

// oml2d.onStageSlideOut(() => {
//   console.log('ssssssssss');
// });
