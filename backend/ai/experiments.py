EXPERIMENTS = []
# EXPERIMENTS += [
#     {
#         "name": "L1-campfire",
#         "dataset": "VFX/stylized_flame/campfire01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "L1-fireball",
#         "dataset": "VFX/stylized_flame/fireball01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "L1-flame",
#         "dataset": "VFX/stylized_flame/flame01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "AnimationTest",
#         "dataset": "VFX/DemoAnim",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 32,
#             "film_time_channels": 8,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "hidden_dim": 32,
#             "apply_film": [1],
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "AnimationTestBigger",
#         "dataset": "VFX/DemoAnim",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 64,
#             "film_time_channels": 8,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "hidden_dim": 64,
#             "apply_film": [1],
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "AnimationTestBiggest",
#         "dataset": "VFX/DemoAnim",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 8,
#             "trunk_time_channels": 2,
#             "film_pos_channels": 128,
#             "film_time_channels": 8,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "hidden_dim": 64,
#             "apply_film": [1],
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "FireTest",
#         "dataset": "VFX/fire/",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 64,
#             "film_time_channels": 8,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "hidden_dim": 64,
#             "apply_film": [1],
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "TestRun",
#         "dataset": "VFX/firething",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 64,
#             "film_time_channels": 8,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "hidden_dim": 64,
#             "apply_film": [1],
#             "output_channels": 3,
#             "learned_encodings": True,
#         }
#     }
# ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"SirenFilmNoLearn{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 2,
#                 "film_time_channels": 1,
#                 "film_pos_scheme": "sinusoidal",
#                 "film_time_scheme": "sinusoidal",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": False,
#                 "siren_film": True,
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"LearnedFilmNoSiren{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "sinusoidal",
#                 "film_time_scheme": "sinusoidal",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"NoLearnFilmNoSiren{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "sinusoidal",
#                 "film_time_scheme": "sinusoidal",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": False,
#                 "siren_film": False,
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"LearnedMathyEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 64,
#                 "trunk_time_channels": 16,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "mathy",
#                 "film_time_scheme": "mathy",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"SinExpEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "sinexp",
#                 "film_time_scheme": "sinexp",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"CosLogEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "coslog",
#                 "film_time_scheme": "coslog",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"PureSinEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "sin",
#                 "film_time_scheme": "sin",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"NewLearnedMathyAltEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin", "cos", "exp", "log1p"],
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"UnlearnedLessMathyEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "less_mathy",
#                 "film_time_scheme": "less_mathy",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": False,
#                 "siren_film": False,
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"LinearFireCircleBigSinEncodingTest{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "linear",
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"WidePrefilmNarrowTrunkFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 32,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "linear",
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"LogRetestWidePrefilmFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "logarithmic",
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"HelmholtzMSE128HiddenFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "helmholtz",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 128,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "encoding_cycle": ["sin"],
#                 "target_resolution": 512,
#             }
#         }
#     ]

# for i in range(2):
#     EXPERIMENTS += [
#         {
#             "name": f"FullNormedHelmholtzFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "helmholtz",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "encoding_cycle": ["sin"],
#                 "target_resolution": 512,
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"SpecificFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "decoder_type": "Specific",
#                 "device": "cuda:1",
#             }
#         }
#     ]


# EXPERIMENTS += [
#     {
#         "name": f"FireLogTest",
#         "dataset": "VFX/fire_circles",
#         "config": {
#             "decoder_type": "Specific",
#             "device": "cuda:1",
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"NormedSmallBeauty",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "Specific",
#             "device": "cuda:1",
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"WeirdTestBeauty",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "Test",
#             "device": "cuda:0",
#         }
#     }
# ]


# EXPERIMENTS += [
#     {
#         "name": f"TConditionedFire",
#         "dataset": "VFX/fire_circles",
#         "config": {
#             "decoder_type": "TLearnedWaveVector",
#             "device": "cuda:1",
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"TimeBaseFire",
#         "dataset": "VFX/fire_circles",
#         "config": {
#             "decoder_type": "Test",
#             "device": "cuda:1",
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"TModulatedBeauty2",
#         "dataset": "VFX/fire_circles",
#         "config": {
#             "decoder_type": "TModulated",
#             "device": "cuda:1",
#         }
#     }
# ]


# EXPERIMENTS += [
#     {
#         "name": f"FourInitTModulatedFire",
#         "dataset": "VFX/fire_circles",
#         "config": {
#             "decoder_type": "TModulated",
#             "device": "cuda",
#             "pos_encoding_len": 256,
#         }
#     }
# ]


# EXPERIMENTS += [
#     {
#         "name": f"FourTModulatedBeauty",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "TModulated",
#             "device": "cuda:1",
#             "pos_encoding_len": 256,
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"WideTimeTest",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "Test",
#             "device": "cuda:1",
#             "time_encoding_len": 1024,
#         }
#     }
# ]

EXPERIMENTS += [
    {
        "name": "35Forever",
        "dataset": "benchmarks/uvg/beauty",
        "config": {
            "decoder_type": "Dual",
            "device": "cuda:1",
            "batch_size": 128,
            "epochs": 100000,
        }
    }
]

# EXPERIMENTS += [
#     {
#         "name": f"WideFourModulatedFire",
#         "dataset": "VFX/fire_circles",
#         "config": {
#             "decoder_type": "TModulated",
#             "device": "cuda",
#             "pos_encoding_len": 1024,
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"TConditionedBeauty",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "TLearnedWaveVector",
#             "device": "cuda",
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"BoundCoordFlowishBeautyTest",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "CoordFlowish",
#             "device": "cuda",
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": f"BigBeauty",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "decoder_type": "Big",
#             "device": "cuda",
#         }
#     }
# ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"SinCosPSNRFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "encoding_cycle": ["sin", "cos"],
#                 "target_resolution": 512,
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"PureMSEFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "linear",
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"TopWidePrefilmFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 512,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "linear",
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"DoubleWidePrefilmFireCircle{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 512,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 128,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "linear",
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"ExponentialFireCircleBigSinEncodingTest{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "exponential",
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"NewInverseFireCircleBigSinEncodingTest{i}",
#             "dataset": "VFX/fire_circles",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "analytic",
#                 "film_time_scheme": "analytic",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#                 "encoding_cycle": ["sin"],
#                 "frequency_initialization": "inverse",
#             }
#         }
#     ]


# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"WidePreFilmPureSinEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "sin",
#                 "film_time_scheme": "sin",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": True,
#                 "siren_film": False,
#             }
#         }
#     ]

# for i in range(3):
#         EXPERIMENTS += [
#         {
#             "name": f"UnlearnedWidePreFilmPureSinEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "sin",
#                 "film_time_scheme": "sin",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": False,
#                 "siren_film": False,
#             }
#         }
#     ]

# for i in range(3):
#         EXPERIMENTS += [
#         {
#             "name": f"LatentUnlearnedWidePreFilmPureSinEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 512,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 256,
#                 "film_time_channels": 64,
#                 "film_pos_scheme": "sin",
#                 "film_time_scheme": "sin",
#                 "film_pos_include_raw": True,
#                 "film_time_include_raw": True,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": False,
#                 "siren_film": False,
#             }
#         }
#     ]

# for i in range(3):
#     EXPERIMENTS += [
#         {
#             "name": f"UnlearnedMathyEncodingTest{i}",
#             "dataset": "VFX/firething",
#             "config": {
#                 "latent_dim": 0,
#                 "trunk_pos_channels": 2,
#                 "trunk_time_channels": 1,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "mathy",
#                 "film_time_scheme": "mathy",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,
#                 "apply_film": [1],
#                 "output_channels": 3,
#                 "learned_encodings": False,
#                 "siren_film": False,
#             }
#         }
#     ]


# EXPERIMENTS += [
#     {
#         "name": "TestSirenNoFilm",
#         "dataset": "VFX/firething",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 0,
#             "film_time_channels": 0,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 0,
#             "hidden_dim": 64,
#             "apply_film": [],
#             "output_channels": 3,
#             "siren_trunk": True,
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "Benchmark1",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 64,
#             "film_time_channels": 8,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "hidden_dim": 64,
#             "apply_film": [1],
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "Big-Benchmark1",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 64,
#             "trunk_time_channels": 16,
#             "film_pos_channels": 256,
#             "film_time_channels": 48,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 128,
#             "hidden_dim": 256,
#             "apply_film": [1],
#             "output_channels": 3,
#         }
#     }
# ]


# EXPERIMENTS += [
#     {
#         "name": "Med-Benchmark1",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 64,
#             "trunk_time_channels": 16,
#             "film_pos_channels": 256,
#             "film_time_channels": 48,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 128,
#             "hidden_dim": 256,
#             "apply_film": [1],
#             "output_channels": 3,
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "LearnedEncoding-Benchmark1",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 64,
#             "trunk_time_channels": 16,
#             "film_pos_channels": 512,
#             "film_time_channels": 64,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 128,
#             "hidden_dim": 256,
#             "apply_film": [1],
#             "output_channels": 3,
#             "learned_encodings": True,
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "WideEncoding-Benchmark1",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 1024,
#             "film_time_channels": 256,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 256,
#             "hidden_dim": 128,
#             "apply_film": [1],
#             "output_channels": 3,
#             "learned_encodings": True,
#         }
#     }
# ]


# EXPERIMENTS += [
#     {
#         "name": "NewEncoding-Benchmark1",
#         "dataset": "benchmarks/uvg/beauty",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 256,
#             "trunk_time_channels": 64,
#             "film_pos_channels": 1024,
#             "film_time_channels": 256,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 128,
#             "hidden_dim": 512,
#             "apply_film": [1],
#             "num_layers": 6,
#             "layer_sizes": [2, 1.5, 1, 0.5, 1, 2],
#             "output_channels": 3,
#             "learned_encodings": True,
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "ossim-small-L1-campfire",
#         "dataset": "VFX/stylized_flame/campfire01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "hidden_dim": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "ossim-small-L1-fireball",
#         "dataset": "VFX/stylized_flame/fireball01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "hidden_dim": 32,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [
#     {
#         "name": "ossim-small-L1-flame",
#         "dataset": "VFX/stylized_flame/flame01",
#         "config": {
#             "latent_dim": 0,
#             "trunk_pos_channels": 2,
#             "trunk_time_channels": 1,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "hidden_dim": 32,
#             "apply_film": [1]
#         }
#     },
# ]

#     {
#         "name": "spiral+raw-trunk-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 8,
#             "trunk_pos_scheme": "sinusoidal",
#             "trunk_pos_include_raw": True,
#             "trunk_time_channels": 4,
#             "trunk_time_scheme": "sinusoidal",
#             "trunk_time_include_raw": True,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-raw-encoded-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-noproj-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-multifilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1, 2, 3]
#         }
#     },
#     {
#         "name": "spiral-singleseed1-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-singleseed2-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     }
# ]

# EXPERIMENTS += [
#     {
#         "name": "spiral-pureraw-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 2,
#             "film_time_channels": 1,
#             "film_pos_scheme": None,
#             "film_time_scheme": None,
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-densefilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-deepfilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1, 2]
#         }
#     },
#     {
#         "name": "spiral-delayedfilm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [2]
#         }
#     },
#     {
#         "name": "spiral-prefilm16-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 16,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-prefilm64-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-trunknormonly-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 2,
#             "trunk_pos_scheme": None,
#             "trunk_pos_include_raw": False,
#             "trunk_pos_include_norm": True,
#             "trunk_time_channels": 1,
#             "trunk_time_scheme": None,
#             "trunk_time_include_raw": False,
#             "trunk_time_include_norm": True,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-nonorm-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "film_pos_include_norm": False,
#             "film_time_include_norm": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-singleseed3-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 16,
#             "film_time_channels": 8,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 32,
#             "apply_film": [1]
#         }
#     }
# ]

# for seed in range(1, 4):
#     EXPERIMENTS += [
#         {
#             "name": f"spiral-shadersmall-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "spiral",
#                 "film_time_scheme": "spiral",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 48,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-shaderdenseproj-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 32,
#                 "film_pos_scheme": "spiral",
#                 "film_time_scheme": "spiral",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 16,  # narrow projection
#                 "hidden_dim": 32,
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-shaderwideencproj64-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 64,
#                 "film_time_channels": 32,
#                 "film_pos_scheme": "spiral",
#                 "film_time_scheme": "spiral",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 64,
#                 "hidden_dim": 32,
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-gaussianfilm-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "gaussian",
#                 "film_time_scheme": "gaussian",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-linearfilm-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "linear",
#                 "film_time_scheme": "linear",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#         {
#             "name": f"spiral-polyfilm-cloud-seed{seed}",
#             "dataset": "VFX/clouds/cloud01",
#             "config": {
#                 "trunk_pos_channels": 0,
#                 "trunk_time_channels": 0,
#                 "film_pos_channels": 32,
#                 "film_time_channels": 16,
#                 "film_pos_scheme": "polynomial",
#                 "film_time_scheme": "polynomial",
#                 "film_pos_include_raw": False,
#                 "film_time_include_raw": False,
#                 "prefilm_dims": 32,
#                 "hidden_dim": 64,  # trunk hidden size
#                 "apply_film": [1]
#             }
#         },
#     ]

# EXPERIMENTS += [
#     {
#         "name": "spiral-noproj-wideinput-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "sinus-proj64-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "sinusoidal",
#             "film_time_scheme": "sinusoidal",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1]
#         }
#     },
#     {
#         "name": "spiral-smallraw-noproj-cloud",
#         "dataset": "VFX/clouds/cloud01",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 4,
#             "film_time_channels": 2,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": True,
#             "film_time_include_raw": True,
#             "prefilm_dims": 0,
#             "apply_film": [1]
#         }
#     },
# ]

# EXPERIMENTS += [{
#         "name": "spiral-densefilm-5mmnist",
#         "dataset": "moving_mnist",
#         "config": {
#             "trunk_pos_channels": 0,
#             "trunk_time_channels": 0,
#             "film_pos_channels": 32,
#             "film_time_channels": 16,
#             "film_pos_scheme": "spiral",
#             "film_time_scheme": "spiral",
#             "film_pos_include_raw": False,
#             "film_time_include_raw": False,
#             "prefilm_dims": 64,
#             "apply_film": [1],
#             "output_channels": 1,
#         }
#     },
# ]

# EXPERIMENTS += [{
#     "model_type": "drill",
#     "name": "sloss-drill-fiber-cloud",
#     "dataset": "VFX/clouds/cloud01",
#     "config": {},
# }]

# EXPERIMENTS += [{
#     "model_type": "drill",
#     "name": "soapdrill-fiber-explosion",
#     "dataset": "VFX/explosions/explosion00",
#     "config": {},
# }]

# EXPERIMENTS += [{
#     "model_type": "drill",
#     "name": "soapdrill-fiber-flame",
#     "dataset": "VFX/hollow-flame",
#     "config": {},
# }]
