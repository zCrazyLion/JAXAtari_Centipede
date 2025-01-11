## Sprite Behaviors
### Player Sub
Each key frame takes 4 frames
### Enemy Sub
Each key frame takes 4 frames
### Enemy Shark
key frame 1 takes 16 frames, key frame 2 takes 8 frames
### Diver 
key frame 1 takes 16 frames, key frame 2 takes 8 frames

### Background
Each key frame takes 8 frames
### Death Animation
First flash b&w, (32 frames) b&w 4 frames each w/ global counter?
death1: 8
death2: 8
death3: 4
death4: 4
death5: 4
death6: 4
death7: 4

## Game State
This introduces the different fields of game state objects.

Remember: while rendering, the order of x and y coordinates should be exchanged for reasons in JAX and PyGame.

### Player Position
`player_x` and `player_y` are JAX singleton arrays that can be converted to `int` type via `.item()` method.

```
player_x=Array(76, dtype=int32, weak_type=True), player_y=Array(46, dtype=int32, weak_type=True)
```
### Player Direction

Player direction is [0] if left, [1] if right. We will omit [] notation for JAX singleton arrays since here.

```
player_direction=Array(0, dtype=int32, weak_type=True), oxygen=Array(1, dtype=int32, weak_type=True), 
```

### Collected Divers

```
divers_collected=Array(0, dtype=int32, weak_type=True)
```

# Score
```
score=Array(0, dtype=int32, weak_type=True), 
```

### Lives

```
lives=Array(3, dtype=int32, weak_type=True)
```

### Diver positions

```
 diver_positions=Array([[160.,  69.,  -1.],
       [160.,  93.,  -1.],
       [160., 117.,  -1.],
       [160., 141.,  -1.]], dtype=float32), 
```       

### Shark Positions
```       
shark_positions=Array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32),
```       

### Enemy Submarine Positions
```
sub_positions=Array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32), enemy_missile_positions=Array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]], dtype=float32),
```

### Enemy Surface Sub Position
```
 surface_sub_position=Array([[0., 0., 0.]], dtype=float32),
``` 

### Player Torpedo Position
```  
  player_missile_position=Array([0., 0., 0.], dtype=float32),
```

### Step Counter
```
   step_counter=Array(1, dtype=int32, weak_type=True),
```





### Unknown

```
    just_surfaced=Array(-1, dtype=int32, weak_type=True)
```

```
 spawn_state=SpawnState(difficulty=Array(0, dtype=int32, weak_type=True), 

 obstacle_pattern_indexes=Array([0, 1, 2, 3], dtype=int32), 
 
 obstacle_attributes=Array([8, 0, 8, 0], dtype=int32), 
 
 spawn_timers=Array([ 0, 30, 60, 90], dtype=int32)), 
 
 ```

## HUD Elements (Needs Verification)
### Score 
Position: (10,10)

### life Indicator
Position: (WINDOW_WIDTH - 200, 10)
### Diver Indicator
Position: (WINDOW_WIDTH - 100, 40)
