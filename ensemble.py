import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Ensemble (Weighted Average) of 3 Models')
    
    # 1. 입력 파일 경로 설정 (기본값 설정)
    parser.add_argument('--deepfm', type=str, default='submission4.csv', help='Path to DeepFM submission csv')
    parser.add_argument('--cb', type=str, default='submission2.csv', help='Path to CB submission csv')
    parser.add_argument('--hy', type=str, default='submission3.csv', help='Path to HY submission csv')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output filename')
    
    # 2. 가중치 설정 (합이 1.0이 되도록 설정하는 것을 권장)
    # 성능이 좋은 모델에 더 높은 가중치를 줍니다. (예: DeepFM > Hybrid > CF)
    parser.add_argument('--w1', type=float, default=0.5, help='Weight for DeepFM')
    parser.add_argument('--w2', type=float, default=0.2, help='Weight for CB')
    parser.add_argument('--w3', type=float, default=0.3, help='Weight for HY')
    
    args = parser.parse_args()

    print("="*50)
    print(f"Ensembling 3 Models...")
    print(f"1. DeepFM ({args.w1}): {args.deepfm}")
    print(f"2. CB ({args.w2}): {args.cb}")
    print(f"3. HY     ({args.w3}): {args.hy}")
    print("="*50)

    # 파일 존재 여부 확인
    for fpath in [args.deepfm, args.cb, args.hy]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

    # 3. 데이터 로드
    df1 = pd.read_csv(args.deepfm)
    df2 = pd.read_csv(args.cb)
    df3 = pd.read_csv(args.hy)

    # 4. 데이터 정합성 체크 (매우 중요)
    # 모든 파일의 행 개수가 같아야 하고, ID 순서가 동일해야 엉뚱한 점수를 섞지 않음
    if len(df1) != len(df2) or len(df1) != len(df3):
        raise ValueError("Error: Submission files have different lengths!")

    # rId 혹은 userId/movieId 기준으로 정렬되어 있는지 확인 (다르면 강제 정렬)
    # 일반적으로 rId가 Key라고 가정
    if 'rId' in df1.columns:
        key_col = 'rId'
    else:
        # rId가 없으면 userId, movieId 복합키라고 가정하고 정렬 수행
        key_col = ['userId', 'movieId']
        for df in [df1, df2, df3]:
            df.sort_values(by=key_col, inplace=True)
            df.reset_index(drop=True, inplace=True)

    # ID 컬럼이 서로 같은지 이중 체크
    if 'rId' in df1.columns:
        if not (df1['rId'].equals(df2['rId']) and df1['rId'].equals(df3['rId'])):
            raise ValueError("Error: IDs in submission files do not match! Please check sort order.")

    # 5. 가중 평균 계산 (Weighted Average)
    # rating 컬럼 추출
    pred1 = df1['rating'].values
    pred2 = df2['rating'].values
    pred3 = df3['rating'].values

    # 가중합
    final_pred = (pred1 * args.w1) + (pred2 * args.w2) + (pred3 * args.w3)
    
    # 가중치 합으로 나누기 (가중치 합이 1이 아닐 경우를 대비해 스케일 보정)
    weight_sum = args.w1 + args.w2 + args.w3
    final_pred = final_pred / weight_sum

    # 6. 후처리 (Clipping & Rounding)
    # (1) 범위 제한: 0.5 ~ 5.0 (데이터셋의 최소/최대 평점에 맞게 수정 가능)
    final_pred = np.clip(final_pred, 0.5, 5.0)

    # (2) 0.5 단위 반올림 (선택사항: 실수 제출이 가능하다면 이 줄 주석 처리)
    final_pred = np.round(final_pred * 2) / 2

    # 7. 결과 저장
    submission = df1.copy() # 첫 번째 파일의 형식을 그대로 복사
    submission['rating'] = final_pred

    submission.to_csv(args.output, index=False)
    print(f"Done! Ensemble result saved to: {args.output}")
    print(f"Sample predictions:\n{submission.head()}")

if __name__ == "__main__":
    main()