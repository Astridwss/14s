#ifndef DATA_STRUCT_INCLUDE_MACRO
#define DATA_STRUCT_INCLUDE_MACRO

//================================无效值定义===============================//
#define MINUS_VALUE -1
#define FOUR_SIGNED_INVALID 0x80000000		//四字节无符号型无效值为最大值,有符号性无效值为-2147483648
#define FOUR_UNSIGNED_INVALID 0xffffffff	//四字节无符号型无效值为最大值
#define DOUBLE_INVALID -9999999.0			//double 类型无效值
#define FLOAT_INVALID -9999999.0			//float 类型无效值
#define ALTITUDE_INVALID -9999999			//高度字段无效值

//强化学习训练中态势信息推送前台
#define PR_ID_RL_TRAINING_SITUATION_TO_FRONT 12290

#endif//DATA_STRUCT_INCLUDE 