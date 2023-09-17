#define rendered texture0
#define depthmap texture1
#define normalmap texture2
#define water_mask texture3

uniform sampler2D rendered;
uniform sampler2D bloom;
uniform sampler2D depthmap;
uniform sampler2D normalmap;
uniform sampler2D water_mask;

uniform vec2 texelSize0;

uniform vec4 skyBgColor;

uniform mat4 mCameraView;
uniform mat4 mCameraViewInv;
uniform mat4 mCameraViewProj;
uniform mat4 mCameraViewProjInv;

uniform vec3 cameraPosition;
uniform vec3 cameraOffset;

uniform float animationTimer;

#ifdef GL_ES
varying mediump vec2 varTexCoord;
#else
centroid varying vec2 varTexCoord;
#endif

float noise(vec3 uvd) {
	return fract(dot(sin(uvd * vec3(13041.19699, 27723.29171, 61029.77801)), vec3(73137.11101, 37312.92319, 10108.89991)));
}

vec2 projectPos(vec3 pos) {
    vec4 projected = mCameraViewProj * vec4(pos, 1.0);
    return (projected.xy / projected.w) * 0.5 + 0.5;
}

vec3 worldPos(vec2 pos) {
    vec4 position = mCameraViewProjInv * vec4((pos - 0.5) * 2.0, texture2D(depthmap, pos).x, 1.0);
    return position.xyz / position.w;
}

vec4 perm(vec4 x)
{
	return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float snoise(vec3 p)
{
	vec3 a = floor(p);
	vec3 d = p - a;
	d = d * d * (3.0 - 2.0 * d);

	vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
	vec4 k1 = perm(b.xyxy);
	vec4 k2 = perm(k1.xyxy + b.zzww);

	vec4 c = k2 + a.zzzz;
	vec4 k3 = perm(c);
	vec4 k4 = perm(c + 1.0);

	vec4 o1 = fract(k3 * (1.0 / 41.0));
	vec4 o2 = fract(k4 * (1.0 / 41.0));

	vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
	vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

	return o4.y * d.y + o4.x * (1.0 - d.y);
}

const float _MaxDistance = 100000.0;
const float _Step = 0.05;
const float _Thickness = 0.05;

void main(void) {
	vec2 uv = varTexCoord.st;
    vec2 render_uv = uv;
	vec4 mask = texture2D(water_mask, uv);
	// vec4 color = vec4(0.0);
    vec4 color = texture2D(rendered, uv);

    // if (mask == vec4(1.0)) { // This somehow catches the sun color ........... somehow
    if (mask == vec4(1.0, 0.0, 1.0, 1.0)) {
        color = skyBgColor;

        vec3 position = worldPos(uv);
        vec3 normal = normalize(mat3(mCameraView) * texture2D(normalmap, uv).xyz);
        vec4 real_pos = vec4(position, 1.0) * mCameraViewInv;
        // normal.x += snoise((position * mat3(mCameraViewInv)) / 10) ;
        // normal.x += (snoise((position * mat3(mCameraView) / 16 + animationTimer * 100) ) * 0.1) - 0.05;

	// vec3 wavePos = (mWorld * pos).xyz + cameraOffset;

	// wavePos.x /= WATER_WAVE_LENGTH * 3.0;
	// wavePos.z /= WATER_WAVE_LENGTH * 2.0;
	// wavePos.z += animationTimer * WATER_WAVE_SPEED * 10.0;
	// pos.y += (snoise(wavePos) - 1.0) * WATER_WAVE_HEIGHT * 5.0;

        // normal.x += (snoise((position * mat3(mCameraView) + cameraPosition)));
        normal.x += (snoise((vec3(position.x / 20, position.y, position.z / 20 + animationTimer * 100) * mat3(mCameraView)))) / 20;

        vec3 ray_step = reflect(normalize(position), normal);
        float ray_length = length(position) * 0.05;
        float start_depth = texture2D(depthmap, uv).x;
        // float ray_length = _Step;

        float stp = _Step;
        vec3 march_position = position;

        vec2 sample_uv;
        float screen_depth, target_depth;

        while (ray_length < _MaxDistance) {
            march_position = position + ray_step * ray_length;
            sample_uv = projectPos(march_position);

            if (sample_uv.x > 1 || sample_uv.x < 0 || sample_uv.y > 1 || sample_uv.y < 0) {
                break;
            }

            screen_depth = worldPos(sample_uv).z;
            target_depth = march_position.z;

            // if ((screen_depth - target_depth) < 0.0005) {
            if (texture2D(depthmap, sample_uv).x > start_depth && march_position.z / screen_depth > 1.01) {
                color = texture2D(rendered, sample_uv);
                // render_uv = sample_uv;
                break;
            }

            ray_length += stp;
            stp *= 1.01;
        }
    }

    gl_FragColor = color;
}
