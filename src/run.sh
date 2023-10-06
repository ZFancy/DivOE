methods=(pretrained oe_tune)
data_models=(cifar10_wrn cifar100_wrn)
gpu=0

if [ "$2" = "0" ] || [ "$2" = "1" ]; then
  if [ "$1" = "MSP" ] || [ "$1" = "energy" ]; then
      for dm in ${data_models[$2]}; do
          array=(${dm//_/ })
          data=${array[0]}
          model=${array[1]}
          load_path="./snapshots/pretrained"
          for method in ${methods[0]}; do
              # MSP with in-distribution samples as pos
              echo "-----------"${dm}_${method}" "$1" score-----------------"
              CUDA_VISIBLE_DEVICES=$gpu python test.py --method_name ${dm}_${method} --num_to_avg 10 --score $1 \
              --load $load_path
          done
      done
      echo "||||||||done with "${dm}_${method}" above |||||||||||||||||||"
  elif [ "$1" = "oe_tune" ] || [ "$1" = "energy_tune" ]; then # fine-tuning
      score=MSP
      if [ "$1" = "energy_tune" ]; then # fine-tuning
          score=energy
      fi
      for dm in ${data_models[$2]}; do
    array=(${dm//_/ })
    data=${array[0]}
    model=${array[1]}
          for seed in 1; do
        echo "---Training with dataset: "$data"---model used:"$model"---seed: "$seed"---score used:"$score"---------"
              if [ "$2" = "0" ]; then
      m_out=-5
      m_in=-23
              elif [ "$2" = "1" ]; then
      m_out=-5
      m_in=-27
        fi
        save_path="./snapshots/${1}/${data}"
        if [ ! -d $save_path ];then
         mkdir -p $save_path
      fi
                echo "---------------"$m_in"------"$m_out"--------------------"
                CUDA_VISIBLE_DEVICES=$gpu python train.py $data --model $model --score $score --seed $seed \
                --m_in $m_in --m_out $m_out --save $save_path
                CUDA_VISIBLE_DEVICES=$gpu python test.py --method_name ${dm}_s${seed}_"$1" \
                --num_to_avg 10 --score $score --load $save_path
          done
      done
      echo "||||||||done with training above "$1"|||||||||||||||||||"
  elif [ "$1" = "MSP_DivOE" ] || [ "$1" = "energy_DivOE" ]; then # fine-tuning
      extrapolation_score=MSP
      score=MSP
      if [ "$1" = "energy_DivOE" ]; then # fine-tuning
          extrapolation_score=energy
      fi
      for dm in ${data_models[$2]}; do
    array=(${dm//_/ })
    data=${array[0]}
    model=${array[1]}
          for seed in 1; do
        echo "---Training with dataset: "$data"---model used:"$model"---seed: "$seed"---score used:"$score"---------"
              if [ "$2" = "0" ]; then
      m_out=-5
      m_in=-23
              elif [ "$2" = "1" ]; then
      m_out=-5
      m_in=-27
        fi
        save_path="./snapshots/${1}/${data}"
        if [ ! -d $save_path ];then
         mkdir -p $save_path
      fi
                echo "---------------"$m_in"------"$m_out"--------------------"
                CUDA_VISIBLE_DEVICES=$gpu python train_DivOE.py $data --model $model --score $score --seed $seed \
                --m_in $m_in --m_out $m_out --save $save_path --extrapolation_ratio $3 --epsilon $4 \
                 --rel_step_size $5 --num_steps $6 --extrapolation_score $extrapolation_score
                CUDA_VISIBLE_DEVICES=$gpu python test.py --method_name ${dm}_s${seed}_"$1" \
                --num_to_avg 10 --score $score --load $save_path
          done
      done
      echo "||||||||done with training above "$1"|||||||||||||||||||"
  fi
elif [ "$2" = "2" ]; then
  if [ "$1" = "oe_tune" ] || [ "$1" = "energy_tune" ]; then # fine-tuning
      score=MSP
      if [ "$1" = "energy_tune" ]; then # fine-tuning
          score=energy
      fi
        save_path="./snapshots/${1}/ImageNet"
        if [ ! -d $save_path ];then
         mkdir -p $save_path
      fi
                CUDA_VISIBLE_DEVICES=$gpu python main_ImageNet.py --score $score --save $save_path
      echo "||||||||done with training above "$1"|||||||||||||||||||"
  elif [ "$1" = "MSP_DivOE" ] || [ "$1" = "energy_DivOE" ]; then # fine-tuning
      extrapolation_score=MSP
      score=MSP
      if [ "$1" = "energy_DivOE" ]; then # fine-tuning
          extrapolation_score=energy
      fi
        save_path="./snapshots/${1}/ImageNet"
        if [ ! -d $save_path ];then
         mkdir -p $save_path
      fi
                CUDA_VISIBLE_DEVICES=$gpu python main_ImageNet_DivOE.py --score $score \
                --save $save_path --extrapolation_ratio $3 --epsilon $4 \
                 --rel_step_size $5 --num_steps $6 --extrapolation_score $extrapolation_score
      echo "||||||||done with training above "$1"|||||||||||||||||||"
  fi
fi