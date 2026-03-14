import mlflow

from utils import get_databricks_user

def register_prompt(prompt_name: str, template: str, commit_message: str | None = None) -> bool:
    

    user_template = template

    few_shot_template = """---

    Few-shot Examples:

    Example 1:
    {{example1}}

    ---

    Example 2:
    {{example2}}

    ---
    Example 3:
    {{example3}}"""

    if "few-shot" in prompt_name:
        user_template = user_template + few_shot_template

    print(user_template)

    commit_message = commit_message or 'Initial Commit'

    print('prompt name: ', prompt_name)

    # register prompt
    prompt = mlflow.genai.register_prompt(
        name = prompt_name,
        template = user_template,
        commit_message = commit_message,
        tags = {
            "author" : get_databricks_user(),
            "use_case" : "information-extraction",
            "language" : "en"
        }

    )

    print(f"Created prompt '{prompt.name}' (version {prompt.version})")
    return True

    # if not mlflow.genai.search_prompts(filter_string=f"name='{prompt_name}'"):
    #     mlflow.genai.register_prompt(
    #         name = prompt_name,
    #         template = system_prompt
    #     )
    # else:
    #     print(f"Prompt - {prompt_name} already exists.")


def set_prompt_alias(prompt_name: str, alias: str, version: int) -> bool:

    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias=alias,
        version=version
    )
    print('Prompt alias set successfully.')
    return True

def get_prompt(prompt_name: str, version: int | None = None, alias: str | None = None, format_prompt_to_single_brace: bool = False) -> str:
    if version is None and alias is None:
        raise ValueError("Either version or alias must be specified.")
    if version is not None and alias is not None:
        raise ValueError("Only one of version or alias can be specified.")

    if version:
        uri = f"prompts:/{prompt_name}/{version}"
    else:
        uri = f"prompts:/{prompt_name}@{alias}"

    prompt = mlflow.genai.load_prompt(uri)

    if format_prompt_to_single_brace:
        return prompt.to_single_brace_format()

    return prompt.template







