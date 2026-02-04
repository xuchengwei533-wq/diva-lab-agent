import type { InternalModel } from 'pixi-live2d-display';
import { Live2DModel, SoundManager } from 'pixi-live2d-display';
import { HitAreaFrames } from 'pixi-live2d-display/extra';
import { isNumber } from 'tianjie';

import type { Events } from './events.js';
import { MotionPreloadStrategy, WindowSizeType } from '../constants/index.js';
import type { DefaultOptions, ModelOptions } from '../types/index.js';
import { getWindowSizeType } from '../utils/index.js';

export class Models {
  model?: Live2DModel<InternalModel>; // 当前模型实例
  private currentModelIndex: number = 0;
  private currentClothesIndex: number = 0;
  private hitAreaFrames: HitAreaFrames;

  constructor(
    private options: DefaultOptions,
    private events: Events
  ) {
    this.hitAreaFrames = new HitAreaFrames();
  }

  get modelIndex(): number {
    return this.currentModelIndex;
  }

  set modelIndex(index: number) {
    this.currentModelIndex = index;
  }

  get modelClothesIndex(): number {
    return this.currentClothesIndex;
  }
  set modelClothesIndex(index: number) {
    this.currentClothesIndex = index;
  }

  get currentModelOptions(): ModelOptions {
    return this.options.models[this.modelIndex];
  }

  create(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.events.emit('load', 'loading');

      let modelPath = this.currentModelOptions.path;

      if (Array.isArray(modelPath)) {
        modelPath = this.currentModelOptions.path[this.modelClothesIndex];
      }

      try {
        this.model = Live2DModel.fromSync(modelPath, {
          motionPreload: (this.currentModelOptions.motionPreloadStrategy as MotionPreloadStrategy) || MotionPreloadStrategy.IDLE,
          onError: () => {
            this.events.emit('load', 'fail');
            reject(new Error('模型加载失败'));
          }
        });

        if (!this.model) {
          this.events.emit('load', 'fail');
          reject(new Error('模型创建失败'));
          return;
        }

        // 加载完成
        this.model.on('load', () => {
          this.events.emit('load', 'success');
          resolve();
        });

        // 模型点击区域被点击
        this.model.on('hit', (names: string[]) => {
          this.events.emit('hit', names);
          this.playRandomMotion(names);
        });
      } catch (error) {
        this.events.emit('load', 'fail');
        reject(error);
      }
    });
  }

  // 设置模型
  settingModel(): void {
    switch (getWindowSizeType()) {
      case WindowSizeType.mobile:
        this.setPosition(...(this.currentModelOptions.mobilePosition || []));
        this.setScale(this.currentModelOptions.mobileScale);
        break;
      case WindowSizeType.pc:
        this.setPosition(...(this.currentModelOptions.position || []));
        this.setScale(this.currentModelOptions.scale);
        break;
    }

    // 是否显示模型可点击区域
    if (this.currentModelOptions.showHitAreaFrames) {
      this.addHitAreaFrames();
    }

    // 音量
    if (isNumber(this.currentModelOptions.volume)) {
      SoundManager.volume = this.currentModelOptions.volume;
    }

    // 设置锚点
    this.setAnchor(...(this.currentModelOptions.anchor || []));

    // 旋转角度
    this.setRotation(this.currentModelOptions.rotation);
  }

  /**
   * 添加点击区域
   */
  addHitAreaFrames(): void {
    if (this.model && this.hitAreaFrames) {
      this.model.addChild(this.hitAreaFrames);
    }
  }

  /**
   * 移除点击区域
   */
  removeHitAreaFrames(): void {
    if (this.model) {
      this.model.removeChildren(0);
    }
  }

  // 模型尺寸
  get modelSize(): { width: number; height: number } {
    return {
      width: this.model?.width || 0,
      height: this.model?.height || 0
    };
  }

  /**
   * 设置缩放比例
   * @param x
   * @param y
   */
  setScale(value: number = 0.1): void {
    if (this.model) {
      this.model.scale.set(value, value);
    }
  }

  /**
   * 设置位置
   * @param x
   * @param y
   */
  setPosition(x = 0, y = 0): void {
    if (this.model) {
      this.model.x = x;
      this.model.y = y;
    }
  }

  /**
   * 设置模型旋转角度
   */
  setRotation(value: number = 0): void {
    if (this.model) {
      this.model.rotation = (Math.PI * value) / 180;
    }
  }

  /**
   * 设置模型在舞台中的锚点位置
   */
  setAnchor(x: number = 0, y: number = 0): void {
    if (this.model) {
      this.model.anchor.set(x, y);
    }
  }

  /**
   * 播放动作
   */
  playMotion(motionGroupName: string, index?: number): void {
    console.log(`尝试播放动作: 组=${motionGroupName}, 索引=${index || 0}`);
    
    // 直接调用Live2D模型的motion方法
    // pixi-live2d-display的motion方法会自动处理动作组和索引
    try {
      void this.model?.motion(motionGroupName, index || 0);
      console.log(`动作播放调用成功: ${motionGroupName}`);
    } catch (error) {
      console.error(`播放动作失败:`, error);
    }
  }

  /**
   * 播放随机动作
   */
  playRandomMotion(areaName: string[]): void {
    console.log(`点击区域: ${areaName}`);
    
    // 根据mao模型的配置，动作组有："Idle" 和 ""（空字符串）
    // "Idle" 包含 mtn_01（待机动作）
    // "" 包含 mtn_02, mtn_03, mtn_04, special_01, special_02, special_03
    
    let targetGroup = '';
    let randomIndex = 0;
    
    if (areaName[0].includes('head') || areaName[0].includes('Head')) {
      // 头部点击：播放特殊动作（在空字符串动作组中，索引3-5）
      targetGroup = '';
      randomIndex = Math.floor(Math.random() * 3) + 3; // 3,4,5
    } else if (areaName[0].includes('body') || areaName[0].includes('Body')) {
      // 身体点击：播放常规动作（在空字符串动作组中，索引0-2）
      targetGroup = '';
      randomIndex = Math.floor(Math.random() * 3); // 0,1,2
    } else {
      // 其他区域：随机选择动作组和索引
      const groups = ['Idle', ''];
      targetGroup = groups[Math.floor(Math.random() * groups.length)];
      
      if (targetGroup === 'Idle') {
        randomIndex = 0; // Idle组只有1个动作
      } else {
        randomIndex = Math.floor(Math.random() * 6); // 空字符串组有6个动作
      }
    }
    
    console.log(`随机播放动作: 组=${targetGroup}, 索引=${randomIndex}`);
    this.playMotion(targetGroup, randomIndex);
  }
}
