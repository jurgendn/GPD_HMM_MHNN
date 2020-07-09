# Generalized Poisson Hidden Markov Model

## Các phần chính của chương trình

### Generalized Poisson Distribution

### Hidden Markov Model

#### Forward Algorithm

#### Backward Algorithm

#### Baum - Welch Algorithm

## Quy ước

### Code Layout

1. **Indentation**
   Dùng tab (4 spaces) để phân định các cấp của lệnh.

   ```python
   # Correct:
   
   # Aligned with opening delimiter.
   foo = long_function_name(var_one, var_two,
                            var_three, var_four)
   
   # Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
   def long_function_name(
           var_one, var_two, var_three,
           var_four):
       print(var_one)
   
   # Hanging indents should add a level.
   foo = long_function_name(
       var_one, var_two,
       var_three, var_four)
   ```

   ```python
   # Wrong:
   
   # Arguments on first line forbidden when not using vertical alignment.
   foo = long_function_name(var_one, var_two,
       var_three, var_four)
   
   # Further indentation required as indentation is not distinguishable.
   def long_function_name(
       var_one, var_two, var_three,
       var_four):
       print(var_one)
   ```

   2. **Tabs hay Spaces**
      Theo autopep8, spaces thường được sử dụng hơn, tuy nhiên ở đây nên thống nhất dùng tabs cho nhanh.

   3. **Giới hạn độ dài**
      Mỗi dòng nên được giới hạn ở 79 kí tự. Đặt trong setting của editor. Các editor có thể khác nhau.

   4. **Xuống dòng trước toán tử toán học**
      Đối với biểu thức dài, xuống dòng trước khi đặt toán tử . Minh họa như sau:

      ```python
      # Wrong:
      # operators sit far away from their operands
      income = (gross_wages +
                taxable_interest +
                (dividends - qualified_dividends) -
                ira_deduction -
                student_loan_interest)
      ```

      Nên được viết như sau:

      ```python
      # Correct:
      # easy to match operators with operands
      income = (gross_wages
                + taxable_interest
                + (dividends - qualified_dividends)
                - ira_deduction
                - student_loan_interest)
      ```

   5. **Dòng trắng**

      - Giữa các phương thức cách nhau bởi ***một dòng trống***
      - Giữa các lớp và phương thức cách nhau bởi ***hai dòng trống***
      - Các khối code cách nhau riêng biệt bằng ***một dòng trống***

   6. **Import**

      1. Các mỗi một module nên được viết trên một dòng khác nhau

         ```python
         # Correct:
         import os
         import sys
         ```

         ```pythonpython]
         # Wrong:
         import sys, os
         ```

      2. Đối với các submodules trong một module, có thể được import chung trong một dòng

         ```python
         # Correct:
         from subprocess import Popen, PIPE
         ```

      3. Import ***luôn luôn*** đặt ở đầu file, ngay sau module comments và docstrings.

      4. Import nên được chia thành 3 phần riêng biệt

         - Thư viện chuẩn
         - Thư viện của bên thứ ba
         - Các module tự viết, local modules

      5. **Quotes**
         Các string được đặt trong ```"string"```

### Đặt tên

#### Class

Các class nên được viết hoa mỗi chữ đầu của một word, viết liền.
Ví dụ

```python
# Correct

class CapWords:

    
class PoissonHiddenMarkovModel:
    
    
# Wrong

class cap_word:
    
    
class Poissonhiddenmarkovmodel:
```



#### Biến và hàm

Viết thường, các từ được phân tách bởi dấu _

Tên hàm, biến có ý nghĩa, không đặt những tên không chỉ mục đích, ý nghĩa của hàm, biết.

```python
# Correct

MAX_ITE = 100	# no more than 100 loops
ite				# current iteration


# Wrong

k = 100			# maximum iteration
i				# current iteration
```

Trong trường hợp nhiều biến, bên cạnh việc đặt tên có ý nghĩa, lập một bảng ở ngoài, định dạng markdown ánh xạ mỗi biến với hàm tương ứng trong paper.

#### Arguments của hàm và method

1. Đối với instance method trong class, luôn dùng `self` là argument đầu tiên
2. Đối với class method, dùng `cls` là argument đầu tiên.

#### Tên method và instance variables

1. Sử dụng tên method như đã quy ước với tên hàm, biến.
2. Dùng **một** dấu _ ở trước tên các hàm, các biến private.

#### Hằng số

Hằng số được VIẾT_HOA_TOÀN_BỘ và phân cách bởi dấu _

---

> Các quy tắc khác sẽ được bổ sung trong quá trình thực hiện

