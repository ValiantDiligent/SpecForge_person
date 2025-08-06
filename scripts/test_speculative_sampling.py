#!/usr/bin/env python3
"""
测试投机采样机制
验证 draft model 和 target model 的交互流程
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class SpeculativeSamplingDemo:
    def __init__(self, draft_model_name, target_model_name):
        """
        初始化投机采样演示
        
        Args:
            draft_model_name: 草稿模型名称（小模型）
            target_model_name: 目标模型名称（大模型）
        """
        print("加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        
        # 添加 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        try:
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.target_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("使用相同模型进行演示...")
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.target_model = self.draft_model
        
        print("模型加载完成!")
    
    def generate_draft_candidates(self, input_ids, num_candidates=3, temperature=0.8):
        """
        使用 draft model 生成候选 tokens
        
        Args:
            input_ids: 输入序列
            num_candidates: 候选 token 数量
            temperature: 采样温度
            
        Returns:
            candidate_ids: 候选 token IDs
            draft_probs: draft model 的概率分布
        """
        print(f"\n🎯 Draft Model 生成 {num_candidates} 个候选 tokens...")
        
        with torch.no_grad():
            candidate_ids = []
            draft_probs = []
            current_input = input_ids.clone()
            
            for i in range(num_candidates):
                # Draft model 前向传播
                outputs = self.draft_model(current_input)
                logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits
                
                # 应用温度
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个 token
                next_token = torch.multinomial(probs, 1)
                candidate_ids.append(next_token.item())
                draft_probs.append(probs[next_token].item())
                
                # 更新输入序列
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                
                # 解码并显示
                token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
                print(f"  候选 {i+1}: '{token_text}' (ID: {next_token.item()}, 概率: {probs[next_token].item():.4f})")
        
        return candidate_ids, draft_probs
    
    def verify_with_target(self, input_ids, candidate_ids, draft_probs, temperature=0.8):
        """
        使用 target model 验证候选 tokens
        
        Args:
            input_ids: 原始输入序列
            candidate_ids: draft model 生成的候选 tokens
            draft_probs: draft model 的概率
            temperature: 采样温度
            
        Returns:
            accepted_tokens: 被接受的 tokens
            rejection_point: 拒绝点位置（-1 表示全部接受）
        """
        print(f"\n🔍 Target Model 验证候选 tokens...")
        
        with torch.no_grad():
            accepted_tokens = []
            current_input = input_ids.clone()
            
            for i, candidate_id in enumerate(candidate_ids):
                # Target model 前向传播
                outputs = self.target_model(current_input)
                logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits
                
                # 应用温度
                logits = logits / temperature
                target_probs = F.softmax(logits, dim=-1)
                
                # 获取候选 token 在 target model 中的概率
                target_prob = target_probs[candidate_id].item()
                draft_prob = draft_probs[i]
                
                # 投机采样的接受概率
                accept_prob = min(1.0, target_prob / draft_prob)
                
                # 随机决定是否接受
                if torch.rand(1).item() < accept_prob:
                    # 接受这个 token
                    accepted_tokens.append(candidate_id)
                    current_input = torch.cat([current_input, torch.tensor([[candidate_id]])], dim=1)
                    
                    token_text = self.tokenizer.decode([candidate_id], skip_special_tokens=True)
                    print(f"  ✅ 接受 token {i+1}: '{token_text}' (接受概率: {accept_prob:.4f})")
                    print(f"     Target概率: {target_prob:.4f}, Draft概率: {draft_prob:.4f}")
                else:
                    # 拒绝这个 token
                    token_text = self.tokenizer.decode([candidate_id], skip_special_tokens=True)
                    print(f"  ❌ 拒绝 token {i+1}: '{token_text}' (接受概率: {accept_prob:.4f})")
                    print(f"     Target概率: {target_prob:.4f}, Draft概率: {draft_prob:.4f}")
                    
                    # 从 target model 重新采样
                    adjusted_probs = torch.clamp(target_probs - draft_probs.unsqueeze(0) * target_probs, min=0)
                    if adjusted_probs.sum() > 0:
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        new_token = torch.multinomial(adjusted_probs, 1)
                        accepted_tokens.append(new_token.item())
                        
                        new_token_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                        print(f"  🔄 重新采样: '{new_token_text}' (ID: {new_token.item()})")
                    
                    return accepted_tokens, i  # 返回拒绝点
            
            # 如果所有候选都被接受，target model 会生成额外的 token
            print(f"  🎉 所有 {len(candidate_ids)} 个候选都被接受！")
            
            # Target model 生成第 5 个 token（额外奖励）
            outputs = self.target_model(current_input)
            logits = outputs.logits[0, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            bonus_token = torch.multinomial(probs, 1)
            accepted_tokens.append(bonus_token.item())
            
            bonus_text = self.tokenizer.decode(bonus_token, skip_special_tokens=True)
            print(f"  🎁 奖励 token: '{bonus_text}' (ID: {bonus_token.item()})")
            
            return accepted_tokens, -1  # -1 表示全部接受
    
    def demo_speculative_sampling(self, prompt, num_steps=3, temperature=0.8):
        """
        演示完整的投机采样流程
        
        Args:
            prompt: 输入提示
            num_steps: draft model 的步数
            temperature: 采样温度
        """
        print("=" * 80)
        print("🚀 投机采样演示")
        print("=" * 80)
        print(f"📝 输入提示: '{prompt}'")
        print(f"🎲 Draft steps: {num_steps}")
        print(f"🌡️  Temperature: {temperature}")
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        print(f"📊 输入长度: {input_ids.shape[1]} tokens")
        
        # 第1步：Target model 生成第一个 token
        print(f"\n🎯 Target Model 生成第1个token...")
        with torch.no_grad():
            outputs = self.target_model(input_ids)
            logits = outputs.logits[0, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            first_token = torch.multinomial(probs, 1)
            
            first_text = self.tokenizer.decode(first_token, skip_special_tokens=True)
            print(f"  第1个token: '{first_text}' (ID: {first_token.item()})")
            
            # 更新输入序列
            input_ids = torch.cat([input_ids, first_token.unsqueeze(0)], dim=1)
        
        # 第2步：Draft model 生成候选 tokens（第2到第4个）
        candidate_ids, draft_probs = self.generate_draft_candidates(
            input_ids, num_candidates=num_steps, temperature=temperature
        )
        
        # 第3步：Target model 验证候选 tokens
        accepted_tokens, rejection_point = self.verify_with_target(
            input_ids, candidate_ids, draft_probs, temperature=temperature
        )
        
        # 结果总结
        print(f"\n📋 总结:")
        print(f"  候选 tokens: {len(candidate_ids)}")
        print(f"  接受 tokens: {len(accepted_tokens)}")
        
        if rejection_point == -1:
            print(f"  🎉 所有候选都被接受，还获得了1个奖励token！")
            print(f"  📈 总加速: 生成了{len(accepted_tokens)}个token，只需要{2}次target model调用")
            efficiency = len(accepted_tokens) / 2
        else:
            print(f"  ❌ 在第{rejection_point + 1}个候选处被拒绝")
            print(f"  📈 部分加速: 生成了{len(accepted_tokens)}个token，需要{rejection_point + 2}次target model调用")
            efficiency = len(accepted_tokens) / (rejection_point + 2)
        
        print(f"  ⚡ 效率提升: {efficiency:.2f}x")
        
        # 显示最终生成的文本
        final_tokens = torch.cat([input_ids, torch.tensor([accepted_tokens])], dim=1)
        final_text = self.tokenizer.decode(final_tokens[0], skip_special_tokens=True)
        print(f"  📄 最终文本: '{final_text}'")
        
        return accepted_tokens, rejection_point

def main():
    # 由于需要两个不同大小的模型，这里用一个模型演示原理
    try:
        demo = SpeculativeSamplingDemo(
            draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",    # 小模型作为draft
            target_model_name="Qwen/Qwen2.5-1.5B-Instruct"    # 大模型作为target
        )
    except:
        print("使用本地可用的模型进行演示...")
        demo = SpeculativeSamplingDemo(
            draft_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            target_model_name="Qwen/Qwen2.5-1.5B-Instruct"
        )
    
    # 测试不同的prompt
    test_prompts = [
        "今天天气很好",
        "人工智能的发展",
        "Python编程语言",
    ]
    
    for prompt in test_prompts:
        demo.demo_speculative_sampling(prompt, num_steps=3, temperature=0.8)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()