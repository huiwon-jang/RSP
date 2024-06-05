import os
import argparse
#import ffmpeg
parser = argparse.ArgumentParser()

# Basic.

#default: 0:401 : all dirs
#0:40 / 40:80 / ... / 360:401
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=401)

parser.add_argument('--datadir', type=str, default='/data/kinetics400')

args = parser.parse_args()

root = os.path.join(args.datadir, 'train')

os.makedirs(os.path.join(args.datadir, 'train2'), exist_ok=True)

dirs = ['changing_wheel', 'ice_climbing', 'hoverboarding', 'playing_kickball', 'playing_clarinet',
        'scuba_diving', 'fixing_hair', 'shaking_head', 'using_computer', 'washing_hands',
        'playing_trombone', 'playing_tennis', 'baby_waking_up', 'blowing_nose', 'scrambling_eggs',
        'pumping_gas', 'changing_oil', 'slacklining', 'cooking_chicken', 'making_jewelry',
        'playing_volleyball', 'smoking_hookah', 'dunking_basketball', 'pushing_car', 'texting',
        'cleaning_floor', 'kicking_field_goal', 'shooting_basketball', 'snowmobiling', 'making_tea',
        'kicking_soccer_ball', 'golf_driving', 'braiding_hair', 'playing_organ', 'balloon_blowing',
        'skipping_rope', 'flying_kite', 'counting_money', 'blowing_out_candles', 'peeling_potatoes',
        'deadlifting', 'bobsledding', 'skiing_crosscountry', 'laughing', 'shaving_head',
        'celebrating', 'exercising_with_an_exercise_ball', 'getting_a_tattoo', 'curling_hair', 'opening_bottle',
        'playing_xylophone', 'tap_dancing', 'bending_metal', 'training_dog', 'busking',
        'climbing_ladder', 'drinking_shots', 'sailing', 'swinging_legs', 'diving_cliff',
        'trimming_trees', 'drawing', 'side_kick', 'mopping_floor', 'tying_bow_tie',
        'kitesurfing', 'eating_chips', 'pole_vault', 'applying_cream', 'riding_elephant',
        'feeding_fish', 'assembling_computer', 'shuffling_cards', 'yawning', 'jogging',
        'feeding_goats', 'faceplanting', 'zumba', 'getting_a_haircut', 'clapping',
        'snowkiting', 'blowing_glass', 'playing_paintball', 'tobogganing', 'golf_chipping',
        'riding_mountain_bike', 'breading_or_breadcrumbing', 'holding_snake', 'baking_cookies', 'triple_jump',
        'petting_animal_not_cat', 'shredding_paper', 'waxing_back', 'folding_clothes', 'riding_or_walking_with_horse',
        'watering_plants', 'bandaging', 'playing_trumpet', 'flipping_pancake', 'trimming_or_shaving_beard',
        'playing_monopoly', 'garbage_collecting', 'checking_tires', 'crossing_river', 'springboard_diving',
        'headbutting', 'running_on_treadmill', 'massaging_feet', 'waxing_legs', 'doing_laundry',
        'playing_accordion', 'sharpening_pencil', 'cracking_neck', 'playing_poker', 'reading_book',
        'dancing_ballet', 'cleaning_windows', 'pull_ups', 'playing_cello', 'crawling_baby',
        'playing_bagpipes', 'stretching_leg', 'playing_cymbals', 'tapping_pen', 'dancing_macarena',
        'playing_basketball', 'dying_hair', 'swimming_butterfly_stroke', 'tying_knot_not_on_a_tie', 'riding_scooter',
        'reading_newspaper', 'sticking_tongue_out', 'exercising_arm', 'extinguishing_fire', 'snorkeling',
        'drinking_beer', 'trapezing', 'milking_cow', 'laying_bricks', 'massaging_persons_head',
        'making_a_cake', 'bench_pressing', 'tying_tie', 'plastering', 'playing_piano',
        'chopping_wood', 'playing_drums', 'folding_paper', 'smoking', 'playing_badminton',
        'hugging', 'salsa_dancing', 'tai_chi', 'tickling', 'feeding_birds',
        'playing_guitar', 'cleaning_toilet', 'blowing_leaves', 'high_jump', 'unboxing',
        'hula_hooping', 'cutting_watermelon', 'catching_or_throwing_softball', 'singing', 'contact_juggling',
        'headbanging', 'crying', 'slapping', 'ice_fishing', 'sweeping_floor',
        'grooming_dog', 'eating_burger', 'building_shed', 'eating_cake', 'whistling',
        'parkour', 'juggling_fire', 'passing_American_football_in_game', 'swimming_breast_stroke', 'bowling',
        'situp', 'cleaning_gutters', 'canoeing_or_kayaking', 'riding_a_bike', 'clean_and_jerk',
        'brush_painting', 'playing_saxophone', 'climbing_tree', 'belly_dancing', 'eating_ice_cream',
        'playing_controller', 'pumping_fist', 'stretching_arm', 'playing_cards', 'playing_chess',
        'washing_hair', 'arm_wrestling', 'swimming_backstroke', 'squat', 'motorcycling',
        'shaving_legs', 'air_drumming', 'carving_pumpkin', 'digging', 'setting_table',
        'welding', 'bouncing_on_trampoline', 'brushing_teeth', 'dribbling_basketball', 'massaging_back',
        'cooking_egg', 'ripping_paper', 'skateboarding', 'dining', 'sign_language_interpreting',
        'abseiling', 'beatboxing', 'throwing_discus', 'driving_tractor', 'pushing_cart',
        'eating_hotdog', 'waxing_chest', 'skiing_not_slalom_or_crosscountry', 'tasting_food', 'doing_nails',
        'folding_napkins', 'ski_jumping', 'rock_scissors_paper', 'hopscotch', 'long_jump',
        'ironing', 'water_skiing', 'playing_recorder', 'walking_the_dog', 'riding_unicycle',
        'windsurfing', 'recording_music', 'dancing_charleston', 'surfing_crowd', 'sled_dog_racing',
        'eating_watermelon', 'riding_mule', 'drinking', 'opening_present', 'playing_keyboard',
        'egg_hunting', 'barbequing', 'throwing_ball', 'jetskiing', 'playing_bass_guitar',
        'moving_furniture', 'shot_put', 'massaging_legs', 'cooking_sausages', 'playing_didgeridoo',
        'finger_snapping', 'passing_American_football_not_in_game', 'dodgeball', 'archery', 'drop_kicking',
        'news_anchoring', 'surfing_water', 'sharpening_knives', 'skydiving', 'krumping',
        'spray_painting', 'swinging_on_something', 'strumming_guitar', 'cleaning_pool', 'hitting_baseball',
        'high_kick', 'dancing_gangnam_style', 'shining_shoes', 'vault', 'making_pizza',
        'playing_ice_hockey', 'tango_dancing', 'planting_trees', 'tapping_guitar', 'catching_or_throwing_baseball',
        'cutting_nails', 'making_snowman', 'playing_ukulele', 'skiing_slalom', 'carrying_baby',
        'grooming_horse', 'throwing_axe', 'spraying', 'blasting_sand', 'weaving_basket',
        'waxing_eyebrows', 'riding_camel', 'giving_or_receiving_award', 'drumming_fingers', 'punching_bag',
        'making_a_sandwich', 'robot_dancing', 'unloading_truck', 'filling_eyebrows', 'cheerleading',
        'sniffing', 'hurdling', 'front_raises', 'testifying', 'playing_squash_or_racquetball',
        'catching_fish', 'brushing_hair', 'shearing_sheep', 'playing_harp', 'eating_carrots',
        'hurling_sport', 'disc_golfing', 'making_sushi', 'doing_aerobics', 'hammer_throw',
        'juggling_balls', 'shaking_hands', 'washing_feet', 'waiting_in_line', 'rock_climbing',
        'making_bed', 'picking_fruit', 'tasting_beer', 'frying_vegetables', 'lunge',
        'cutting_pineapple', 'eating_spaghetti', 'mowing_lawn', 'cartwheeling', 'cooking_on_campfire',
        'biking_through_snow', 'wrestling', 'sneezing', 'push_up', 'bending_back',
        'writing', 'building_cabinet', 'driving_car', 'yoga', 'washing_dishes',
        'tossing_coin', 'decorating_the_christmas_tree', 'auctioning', 'roller_skating', 'jumping_into_pool',
        'tossing_salad', 'bartending', 'catching_or_throwing_frisbee', 'stomping_grapes', 'knitting',
        'marching', 'playing_harmonica', 'snatch_weight_lifting', 'bee_keeping', 'ice_skating',
        'gymnastics_tumbling', 'playing_violin', 'hockey_stop', 'gargling', 'spinning_poi',
        'somersaulting', 'peeling_apples', 'swing_dancing', 'presenting_weather_forecast', 'answering_questions',
        'grinding_meat', 'paragliding', 'country_line_dancing', 'jumpstyle_dancing', 'playing_cricket',
        'pushing_wheelchair', 'cleaning_shoes', 'bungee_jumping', 'sanding_floor', 'using_segway',
        'climbing_a_rope', 'clay_pottery_making', 'snowboarding', 'parasailing', 'petting_cat',
        'capoeira', 'golf_putting', 'water_sliding', 'javelin_throw', 'eating_doughnuts',
        'applauding', 'riding_mechanical_bull', 'shoveling_snow', 'taking_a_shower', 'breakdancing',
        'playing_flute', 'juggling_soccer_ball', 'using_remote_controller_not_gaming', 'punching_person_boxing', 'kissing',
        'shooting_goal_soccer', 'arranging_flowers', 'sword_fighting', 'wrapping_present', 'bookbinding',
        'replacement_for_corrupted_k400']

for dir in dirs[args.begin:args.end]:
    os.makedirs(os.path.join(args.datadir, 'train2', dir), exist_ok=True)
    for filename in os.listdir(os.path.join(root, dir)):
        os.system(
            f'''ffmpeg -i {os.path.join(root, dir, filename)}
            -vf "scale='if(gt(ih,iw),256,trunc(oh*a/2)*2):if(gt(ih,iw),trunc(ow/a/2)*2,256)'"
            {os.path.join(args.datadir, 'train2', dir, filename)}'''
        )
