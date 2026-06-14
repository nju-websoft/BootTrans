import re

# --- 步骤 1: 移除 main 函数 (保持不变，这个模式相对稳定) ---
def remove_main_method(java_code: str) -> str:
    """ 使用精确模式移除 main 方法。 """
    main_pattern = re.compile(
        # 匹配方法签名
        r'public\s+static\s+void\s+main\s*\(\s*String\s*\[\s*\]\s*args\s*\)\s*\{'
        # 非贪婪匹配方法体内容
        r'[\s\S]*?'
        # 匹配方法体的结束花括号
        r'\n\s*\}'
        , re.DOTALL
    )
    return main_pattern.sub('', java_code)

# --- 步骤 2: 提取所有非 main 函数 (修正的、更健壮的模式) ---
def extract_non_main_methods(java_code: str) -> str:
    """
    提取所有非 main 方法，移除类/导入声明，并将方法签名替换为 `static <原返回类型> f_filled(...)`。
    """
    # 1. 移除 main 方法
    code_without_main = remove_main_method(java_code)

    # 2. 移除 import 和 class 声明 (简化代码块)
    code_without_header = re.sub(r'^\s*import\s+.*?;', '', code_without_main, flags=re.MULTILINE)
    code_without_header = re.sub(r'public\s+class\s+\w+\s*\{', '', code_without_header)
    code_without_header = re.sub(r'\}\s*$', '', code_without_header.strip())

    # # 3. 提取方法块
    # method_pattern = re.compile(
    #     r'(?:public|protected|private|static|\s)+[^\s\(]+\s+[^\s]+\s*\([^\)]*\)\s*\{[\s\S]*?\n\s*\}',
    #     re.DOTALL
    # )
    # methods = [match.group(0) for match in method_pattern.finditer(code_without_header)]

    # 4. 使用正则将方法签名替换为 static <原返回类型> f_filled(...)
    signature_pattern = re.compile(
        r'^(\s*)(?:public|protected|private|static|\s)+([^\s\(]+\s*(?:<[^>]+>)?(?:\s*\[\])?)\s+[^\s]+\s*\(([^\)]*)\)',
        re.MULTILINE
    )

    def normalize_signature(method_block: str) -> str:
        def _replacer(match: re.Match) -> str:
            indent = match.group(1)
            return_type = match.group(2).strip()
            params = match.group(3).strip()
            return f"{indent}static {return_type} f_filled({params})"

        normalized, replaced = signature_pattern.subn(_replacer, method_block, count=1)
        return normalized if replaced else method_block
    
    ret = normalize_signature(code_without_header)
    return ret
    # final_methods = [normalize_signature(method.strip()) for method in methods]
    # return '\n\n'.join(ret).strip()

# --- 你的 Java 示例代码 ---
java_example_code = """
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt();
        System.out.println(findNth(n));
    }

    public static int findNth(int n) {
        int count = 0;
        for (long curr = 0; ; curr++) {
            int sum = 0;
            long x = curr;
            while (x != 0) {
                sum += x % 10;
                x /= 10;
            }
            if (sum == 10) {
                count++;
            }
            if (count == n) {
                return (int) curr;
            }
        }
        return -1; // This line is unreachable due to the loop structure
    }
}
"""

# 执行提取
extracted_code = extract_non_main_methods(java_example_code)

print("--- 提取结果 (健壮模式) ---")
print(extracted_code)
