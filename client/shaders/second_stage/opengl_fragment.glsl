#define rendered texture0
#define bloom texture1
#define depthmap texture3
#define normalmap texture4
#define water_mask texture5

struct ExposureParams {
	float compensationFactor;
};

uniform sampler2D rendered;
uniform sampler2D bloom;
uniform sampler2D depthmap;
uniform sampler2D normalmap;
uniform sampler2D water_mask;

uniform vec2 texelSize0;

uniform ExposureParams exposureParams;
uniform lowp float bloomIntensity;
uniform lowp float saturation;

uniform vec4 skyBgColor;

uniform mat4 mCameraView;
uniform mat4 mCameraViewInv;
uniform mat4 mCameraViewProj;
uniform mat4 mCameraViewProjInv;

uniform float animationTimer;

#ifdef GL_ES
varying mediump vec2 varTexCoord;
#else
centroid varying vec2 varTexCoord;
#endif

#ifdef ENABLE_AUTO_EXPOSURE
varying float exposure; // linear exposure factor, see vertex shader
#endif

#ifdef ENABLE_BLOOM

vec4 applyBloom(vec4 color, vec2 uv)
{
	vec3 light = texture2D(bloom, uv).rgb;
#ifdef ENABLE_BLOOM_DEBUG
	if (uv.x > 0.5 && uv.y < 0.5)
		return vec4(light, color.a);
	if (uv.x < 0.5)
		return color;
#endif
	color.rgb = mix(color.rgb, light, bloomIntensity);
	return color;
}

#endif

#if ENABLE_TONE_MAPPING

/* Hable's UC2 Tone mapping parameters
	A = 0.22;
	B = 0.30;
	C = 0.10;
	D = 0.20;
	E = 0.01;
	F = 0.30;
	W = 11.2;
	equation used:  ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
*/

vec3 uncharted2Tonemap(vec3 x)
{
	return ((x * (0.22 * x + 0.03) + 0.002) / (x * (0.22 * x + 0.3) + 0.06)) - 0.03333;
}

vec4 applyToneMapping(vec4 color)
{
	const float exposureBias = 2.0;
	color.rgb = uncharted2Tonemap(exposureBias * color.rgb);
	// Precalculated white_scale from
	//vec3 whiteScale = 1.0 / uncharted2Tonemap(vec3(W));
	vec3 whiteScale = vec3(1.036015346);
	color.rgb *= whiteScale;
	return color;
}

vec3 applySaturation(vec3 color, float factor)
{
	// Calculate the perceived luminosity from the RGB color.
	// See also: https://www.w3.org/WAI/GL/wiki/Relative_luminance
	float brightness = dot(color, vec3(0.2125, 0.7154, 0.0721));
	return max(vec3(0.), mix(vec3(brightness), color, factor));
}
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

void main(void)
{
	vec2 uv = varTexCoord.st;
    vec2 render_uv = uv;
	vec4 mask = texture2D(water_mask, uv).rgba;
	vec4 color = vec4(0.0);
    // vec4 color = texture2D(rendered, uv).rgba;

    // if (mask == vec4(1.0)) { // This somehow catches the sun color ........... somehow
    if (mask == vec4(1.0, 0.0, 1.0, 1.0)) {
        color = skyBgColor;

        vec3 position = worldPos(uv);
        vec3 normal = normalize(mat3(mCameraView) * texture2D(normalmap, uv).xyz);
        // normal.x += (snoise((position * mat3(mCameraViewInv)) ) * 0.1) - 0.05;

        vec3 ray_step = reflect(normalize(position), normal);
        float ray_length = length(position) * 0.05;
        // float ray_length = _Step;

        float stp = _Step;
        vec3 march_position = position;

        vec2 sample_uv;
        float screen_depth, target_depth;

        while (ray_length < _MaxDistance) {
            march_position = position + ray_step * ray_length;
            sample_uv = projectPos(march_position);

            screen_depth = worldPos(sample_uv).z;
            target_depth = march_position.z;

            // if ((screen_depth - target_depth) < 0.0005) {
            if (march_position.z / screen_depth > 1.01) {
                // color = texture2D(rendered, sample_uv);
                render_uv = sample_uv;
                break;
            }

            ray_length += stp;
            stp *= 1.01;
        }
    }

#ifdef ENABLE_SSAA
	color = vec4(0.);
	for (float dx = 1.; dx < SSAA_SCALE; dx += 2.)
	for (float dy = 1.; dy < SSAA_SCALE; dy += 2.)
    color += texture2D(rendered, render_uv + texelSize0 * vec2(dx, dy)).rgba;
	color /= SSAA_SCALE * SSAA_SCALE / 4.;
#else
	color = texture2D(rendered, render_uv).rgba;
#endif

	// translate to linear colorspace (approximate)
	color.rgb = pow(color.rgb, vec3(2.2));

#if ENABLE_TONE_MAPPING
	color.rgb = applySaturation(color.rgb, 1.25);
#endif	

#ifdef ENABLE_BLOOM_DEBUG
	if (uv.x > 0.5 || uv.y > 0.5)
#endif
	{
		color.rgb *= exposureParams.compensationFactor;
#ifdef ENABLE_AUTO_EXPOSURE
		color.rgb *= exposure;
#endif
	}


#ifdef ENABLE_BLOOM
	color = applyBloom(color, uv);
#endif

#ifdef ENABLE_BLOOM_DEBUG
	if (uv.x > 0.5 || uv.y > 0.5)
#endif
	{
#if ENABLE_TONE_MAPPING
		color = applyToneMapping(color);
		color.rgb = applySaturation(color.rgb, saturation);
#endif
	}

	color.rgb = clamp(color.rgb, vec3(0.), vec3(1.));

	// return to sRGB colorspace (approximate)
	color.rgb = pow(color.rgb, vec3(1.0 / 2.2));

	gl_FragColor = vec4(color.rgb, 1.0); // force full alpha to avoid holes in the image.
}
