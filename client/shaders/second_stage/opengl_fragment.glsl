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

uniform vec3 cameraPosition;
uniform mat4 mCameraView;
uniform mat4 mCameraViewInv;
uniform mat4 mCameraViewProj;
uniform mat4 mCameraViewProjInv;

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

float det(mat2 matrix) {
    return matrix[0].x * matrix[1].y - matrix[0].y * matrix[1].x;
}

mat3 inverse(mat3 matrix) {
    vec3 row0 = matrix[0];
    vec3 row1 = matrix[1];
    vec3 row2 = matrix[2];

    vec3 minors0 = vec3(
        det(mat2(row1.y, row1.z, row2.y, row2.z)),
        det(mat2(row1.z, row1.x, row2.z, row2.x)),
        det(mat2(row1.x, row1.y, row2.x, row2.y))
    );
    vec3 minors1 = vec3(
        det(mat2(row2.y, row2.z, row0.y, row0.z)),
        det(mat2(row2.z, row2.x, row0.z, row0.x)),
        det(mat2(row2.x, row2.y, row0.x, row0.y))
    );
    vec3 minors2 = vec3(
        det(mat2(row0.y, row0.z, row1.y, row1.z)),
        det(mat2(row0.z, row0.x, row1.z, row1.x)),
        det(mat2(row0.x, row0.y, row1.x, row1.y))
    );

    mat3 adj = transpose(mat3(minors0, minors1, minors2));

    return (1.0 / dot(row0, minors0)) * adj;
}

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Frame {
    vec3 x, y, z;
};

const float _MaxDistance = 100000.0;
const float _Step = 0.05;
const float _Thickness = 0.05;
const float _Bias = 0.05;
const float _Near = 1.0;
const float _Far = 1000.0;

Frame _Camera = Frame(vec3(1, 0, 0), vec3(0, -1, 0), vec3(0, 0, 1));

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

float mapDepth(float depth)
{
	return min(1., 1. / (1.00001 - depth) / 1000.0);
}

float noise(vec3 uvd) {
	return fract(dot(sin(uvd * vec3(13041.19699, 27723.29171, 61029.77801)), vec3(73137.11101, 37312.92319, 10108.89991)));
}

vec2 projectOnScreen(vec3 eye, vec3 point) {
    vec3 toPoint = (point - eye);
    point = (point - toPoint * (1.0 - _Near / dot(toPoint, _Camera.z)));
    point -= eye + _Near * _Camera.z;
    return point.xy;
}

vec3 posFromDepth(vec2 uv, float depth) {
    vec4 ndc = vec4((uv - 0.5) * 2, depth, 1.0);
    vec4 inversed = mCameraViewProjInv * ndc;
    return (inversed / inversed.w).xyz;
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

uniform float animationTimer;

void main(void)
{
	vec2 uv = varTexCoord.st;
	vec4 mask = texture2D(water_mask, uv).rgba;
    vec4 color = texture2D(rendered, uv).rgba;

    // if (mask == vec4(1.0)) { // This somehow catches the sun color ........... somehow
    if (mask == vec4(1.0, 0.0, 1.0, 1.0)) {
        color = skyBgColor;
        vec3 position = worldPos(uv);
        vec3 normal = normalize(mat3(mCameraView) * texture2D(normalmap, uv).xyz);
        // normal.x *= sin((position * mat3(mCameraView)).x);
        // normal.x +=  0.1 * cos(((position * mat3(mCameraView)).x) / 16 + animationTimer * 250);
        // normal.z +=  0.1 * sin(((position * mat3(mCameraView)).z) / 12 + animationTimer * 250);
        // normal.x += noise(position + animationTimer * 100) / 500;
        // normal.x += (snoise((position * mat3(mCameraView) / 16 + animationTimer * 100) ) * 0.1) - 0.05;
        // normal.z += (snoise((position * mat3(mCameraView)) / 32 + animationTimer * 100) * 0.1) - 0.1;
        vec3 dir = reflect(normalize(position), normal);

        // float ray_length = _Step;
        float ray_length = length(position) * _Step;
        float stp = _Step;
        vec3 march_position = position;

        vec2 sample_uv;
        float screen_depth, target_depth;

        while (ray_length < _MaxDistance) {
        // for (int i = 0; i < 1000; i++) {
            march_position = position + dir * ray_length;
            sample_uv = projectPos(march_position);

            screen_depth = worldPos(sample_uv).z;
            target_depth = march_position.z; // WHY???

            // if ((screen_depth - target_depth) < 0.05) {
            // color.rgb = vec3(texture2D(depthmap, sample_uv).x);
            // color.rgb = vec3(screen_depth / 1000);
                // color.rgb = vec3(abs(screen_depth - target_depth));
            // color.rgb = position.xyz;
            if ((screen_depth - target_depth) < 0.0001) {
            // if (march_position.z / screen_depth > 1.05) {
            // if (screen_depth < target_depth) {
                // color = vec4(march_position, 1);
                color = texture2D(rendered, sample_uv);
                break;
            }

            ray_length += stp;
            stp *= 1.01;
        }

        // gl_FragColor = vec4(color.rgb, 1.0);
        // return;
        
        
                //     // vec3 normal = normalize(texture2D(normalmap, uv).xyz);
                //     // vec4 position = mCameraViewProjInv * vec4(uv, texture2D(depthmap, uv).x, 1.0);
                //     // vec3 ray_dir = normalize(reflect(normalize(position.xyz), normal));
                //     // vec3 ray_step = ray_dir * _Step;

                //     // vec2 reflection_uv = uv;
                //     // for (int i = 1; i < _MaxDistance; i++) {
                //     //     position.xyz += ray_step;

                //     //     vec4 projected = mCameraViewProj * vec4(position.xyz, 1.0);
                //     //     projected.xy /= projected.w;
                        
                //     //     vec2 view_position = projected.xy * 0.5 + 0.5;

                //     //     float depth_ray = projected.z / projected.w;
                //     //     float depth_sample = texture2D(depthmap, view_position).x;

                //     //     if (depth_ray - depth_sample > 0 && depth_ray - depth_sample < _Thickness) {
                //     //         reflection_uv = view_position;
                //     //         break;
                //     //     }
                //     // }
                            
                //                 // vec3 position = posFromDepth(uv, texture2D(depthmap, uv).x);
                //                 // vec4 normal = mCameraView * vec4(texture2D(normalmap, uv).xyz, 0.0);
                //                 // vec3 reflected = normalize(reflect(position, normalize(normal.xyz)));
                        
                //                 // // float mdepth = min(1., 1. / (1.00001 - depth) / _Far);
                //                 // // float aspect = texelSize0.y / texelSize0.x;
                //                 // // float depth = texture2D(depthmap, uv).x;
                //                 // // vec3 position = posFromDepth(uv, texture2D(depthmap, uv).x);
                //                 // // vec3 position = (mCameraViewProjInv * vec4(uv, texture2D(depthmap, uv).x, 1.0)).xyz;
                //                 // // vec3 normal = normalize((mCameraView * vec4(texture2D(normalmap, uv).xyz, 0.0)).xyz);
                            
                //                 // // vec3 reflected = normalize(reflect(normalize(position), normal));
                //                 // vec3 ray_step = _Step * reflected;

                //                 // // vec3 camera_dir = normalize(view_pos);
                //                 // // vec3 ray_dir = normalize(reflect(camera_dir, normal_view));

                //                 // vec2 reflection_uv = uv;
                //                 // // vec3 marching_pos = position + ray_step;

                //                 // float delta;
                //                 // float sampled_depth;
                //                 // vec2 screen_pos;
                //                 // // vec3 marching_pos;
                //                 // vec3 march_reflection = position + ray_step;
                //                 // // float current_depth = depth;

                //                 // // for (float i = _Step; i < _MaxDistance; i += _Step) {
                //                 // for (float i = 0; i < 1000; i ++) {
                //                 //     // vec3 march_reflection = position + i * reflected;
                //                 //     // vec2 screen_pos = projectPos(march_reflection);
                //                 //     // vec3 sample_pos = posFromDepth(screen_pos, texture2D(depthmap, screen_pos).x);
                //                 //     // // sampled_depth = texture2D(depthmap, screen_pos).x;

                //                 //     // if (sample_pos.z > march_reflection.z) {
                //                 //     // // if (texture2D(depthmap, screen_pos).x > march_reflection.z / _Far) {
                //                 //     //     reflection_uv = screen_pos;
                //                 //     //     break;
                //                 //     // }
                                    
                                    
                //                 //     // march_reflection = position + i * reflection;
                //                 //     // float target_depth = march_reflection.z / _Far;
                //                 //     // marching_pos = position + reflection * i;
                //                 //     screen_pos = projectPos(march_reflection);
                //                 //     sampled_depth = abs(posFromDepth(screen_pos, texture2D(depthmap, screen_pos).x).z);
                //                 //     delta = abs(march_reflection.z) - sampled_depth;

                //                 //     if (abs(delta) < _Bias) {
                //                 //         reflection_uv = screen_pos;
                //                 //         break;
                //                 //     }

                //                 //     march_reflection += ray_step;

                //                 // }

                //                 // gl_FragColor = vec4(texture2D(rendered, reflection_uv).rgb, 1.0);
                // // gl_FragColor = vec4(normal_view, 1.0);
                // // gl_FragColor = vec4(view_pos, 1.0);
                // // gl_FragColor = vec4(reflected, 1.0);
                // // gl_FragColor = vec4(reflection_uv, 0.0, 1.0);

                // // vec4 pixelPos = _mWorldViewProjInv * (_mWorldView * gl_FragCoord);
                // // gl_FragColor = vec4(normalize(pixelPos.xyz), 1.0);
                // // gl_FragColor = vec4((_mWorldView * vec4(normal, 1.0)).xyz, 1.0);
                // // gl_FragColor = vec4((normal + 1) / 2, 1.0);
                // // gl_FragColor = vec4(transpose(inverse(mat3(_mWorldView))) * normal, 1.0);
                // // gl_FragColor = vec4((normal_scene + 1) / 2, 1.0);
                
                // // // float depth = texture2D(depthmap, uv).x;

                // // vec3 normal = texture2D(normalmap, uv).xyz;
                // // // vec3 normal_scene = _mInverseWorldViewProj * normal;

                // // vec3 sample_clip_pos = vec3(gl_FragCoord.xy * texelSize0.xy * 2 - 1.0, depth);
                // // sample_clip_pos.y *= -1;
                // // vec3 sample_view_pos = _mInverseWorldViewProj * sample_clip_pos;

                // // vec3 sample_dir = normalize(sample_view_pos);
                // // vec3 reflection_dir = vec3(reflect(sample_dir, normal_scene));
                
                // // // Depth at water surface
                // // float currentDepth = depth;
                // // for (float i = _Step; i < _MaxDistance; i += _Step) {

                // // }


                // // gl_FragColor = vec4(-look_dir, 1.0);

                
                //         float aspect = texelSize0.y / texelSize0.x;
                //         // vec3 lookDir = normalize(inverse(mat3(viewMatrix)) * vec3(0, 0, 1));

                //         // View ray
                //         vec2 r_uv = 2.0 * gl_FragCoord.xy / (1 / texelSize0.y) - vec2(aspect, 1.0);
                //         vec3 r_dir = vec3(r_uv.x * _Camera.x + r_uv.y * _Camera.y + _Near * _Camera.z);
                //         // Ray ray = Ray(cameraOffset, normalize(r_dir));

                //         float depth = texture2D(depthmap, uv).x;
                //         // vec3 normal = texture2D(normalmap, uv).xyz;
                //         // vec3 normal_screen = transpose(inverse(mat3(_mWorldView))) * normal;

                //         vec3 view = r_dir * depth * _Far / _Near;
                //         // vec3 reflected = reflect(normalize(view), normal_screen);
                //         // vec3 position = ray.origin + view;

                //         vec3 position = cameraPosition - posFromDepth(uv, texture2D(depthmap, uv).x);
                //         vec4 normal = mCameraView * vec4(texture2D(normalmap, uv).xyz, 0.0);
                //         vec3 reflected = normalize(reflect(position, normalize(normal.xyz)));

                //         vec2 reflectionUV = uv;
                //         float atten = 0.0;

                //         // World-Space March
                //         vec3 marchReflection;
                //         float currentDepth = depth;
                //         for (float i = _Step; i < _MaxDistance; i += _Step) {
                //             marchReflection = i * reflected;
                            
                //             float targetDepth = dot(view + marchReflection, _Camera.z) / _Far;
                //             // vec2 target = projectOnScreen(cameraOffset, position + marchReflection);
                //             vec2 target = (mCameraViewProjInv * vec4(position + marchReflection, 1.0)).xy;

                //             target.x = map(target.x, -aspect, aspect, 0.0, 1.0);
                //             target.y = map(target.y, -1.0, 1.0, 0.0, 1.0);

                //             float sampledDepth = texture2D(depthmap, target).x;
                //             float depthDiff = sampledDepth - currentDepth;

                //             if (depthDiff > 0.0 && depthDiff < targetDepth - currentDepth + _Thickness) {
                //                 reflectionUV = target;
                //                 atten = 1.0 - i / _MaxDistance;
                //                 break;
                //             }

                //             currentDepth = targetDepth;
                //             if (currentDepth > 1.0) {
                //                 atten = 1.0;
                //                 break;
                //             }
                //         }

                // gl_FragColor = vec4(reflected, 1.0);
                // return;

                // // Screen-Space March
                // // vec2 screenStart = (_mWorldViewProj * vec4(position, 1.0)).xy;
                // // vec2 screenEnd = (_mWorldViewProj * vec4(position + reflected, 1.0)).xy;
                // // vec2 screenDir = (screenEnd - screenStart).xy;

                // // float reflectedDepth = dot(reflected, _Camera.z) / _Far;
                // // float depthStep = reflectedDepth;

                // // float currentDepth = depth;
                // // vec2 march = screenStart;

                // // for (float i = 0.0; i < _MaxDistance; i += _Step) {
                // //     march += screenDir * _Step;

                // //     vec2 marchUV;
                // //     marchUV.x = map(march.x, -aspect, aspect, 0.0, 1.0);
                // //     marchUV.y = map(march.y, -1.0, 1.0, 0.0, 1.0);

                // //     float targetDepth = texture2D(depthmap, marchUV).x;
                // //     float depthDiff = targetDepth - currentDepth;

                // //     if (depthDiff > 0.0 && depthDiff < depthStep) {
                // //         reflectionUV = marchUV;
                // //         atten = 1.0 - 1 / _MaxDistance;
                // //         break;
                // //     }

                // //     currentDepth += depthStep * _Step;
                // // }

                // // // vec3 nrm = inverse(mat3(_mWorldView)) * texture2D(normalmap, uv).xyz;
                // // // gl_FragColor = vec4(nrm, 1.0);
                // gl_FragColor = vec4(texture2D(rendered, reflectionUV).rgb * atten, 1.0);

                // return;
    }
    
#ifdef ENABLE_SSAA
	// vec4 color = vec4(0.);
	for (float dx = 1.; dx < SSAA_SCALE; dx += 2.)
	for (float dy = 1.; dy < SSAA_SCALE; dy += 2.)
		// color += texture2D(rendered, uv + texelSize0 * vec2(dx, dy)).rgba;
	color /= SSAA_SCALE * SSAA_SCALE / 4.;
#else
	// vec4 color = texture2D(rendered, uv).rgba;
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
