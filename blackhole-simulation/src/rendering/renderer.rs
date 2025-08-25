use crate::physics::black_hole::BlackHole;
use crate::rendering::camera::Camera;
use crate::rendering::ray_tracer::{RayTracer, RayTracingQuality};
use crate::rendering::shaders::ShaderManager;
use anyhow::Result;
use std::sync::Arc;
use winit::window::Window;

#[derive(Clone, Copy, Debug)]
pub enum RenderQuality {
    Low,
    Medium,
    High,
}

pub struct Renderer<'w> {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'w>,
    pub config: wgpu::SurfaceConfiguration,
    pub shader_manager: ShaderManager,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub ray_tracer: Option<RayTracer>,
    framebuffer: Vec<[f32; 4]>,
    // GPU presentation resources
    fb_texture: wgpu::Texture,
    fb_view: wgpu::TextureView,
    fb_sampler: wgpu::Sampler,
    fb_bind_group_layout: wgpu::BindGroupLayout,
    fb_bind_group: wgpu::BindGroup,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
}

impl<'w> Renderer<'w> {
    pub async fn new(
        window: &'w Window,
        quality: RenderQuality,
        camera: Arc<Camera>,
        bh: Arc<BlackHole>,
    ) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No adapter"))?;
        let limits = wgpu::Limits::default();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await?;
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let shader_manager = ShaderManager::new(device.clone(), queue.clone());
        let rt_quality = match quality {
            RenderQuality::Low => RayTracingQuality::Low,
            RenderQuality::Medium => RayTracingQuality::Medium,
            RenderQuality::High => RayTracingQuality::High,
        };
        let ray_tracer = Some(RayTracer::new(rt_quality, camera, bh));
        let framebuffer = vec![[0.0; 4]; (config.width * config.height) as usize];

        // Load shaders (basic fullscreen blit). Errors ignored if already loaded.
        let mut sm = shader_manager;
        let _ = sm.load_wgsl_file(
            "fullscreen_vs",
            std::path::Path::new("assets/shaders/vertex.wgsl"),
        );
        let _ = sm.load_wgsl_file(
            "fullscreen_fs",
            std::path::Path::new("assets/shaders/fragment.wgsl"),
        );
        let vs = sm
            .get("fullscreen_vs")
            .ok_or_else(|| anyhow::anyhow!("vertex shader missing"))?;
        let fs = sm
            .get("fullscreen_fs")
            .ok_or_else(|| anyhow::anyhow!("fragment shader missing"))?;

        // Texture to upload CPU ray traced buffer (RGBA32F)
        let tex_extent = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let fb_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framebuffer_texture"),
            size: tex_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fb_view = fb_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let fb_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fb_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let fb_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fb_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let fb_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fb_bg"),
            layout: &fb_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&fb_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fb_view),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&fb_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fullscreen_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: vs,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fs,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            device,
            queue,
            surface,
            config,
            shader_manager: sm,
            size,
            ray_tracer,
            framebuffer,
            fb_texture,
            fb_view,
            fb_sampler,
            fb_bind_group_layout,
            fb_bind_group,
            pipeline_layout,
            pipeline,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.framebuffer
            .resize((self.config.width * self.config.height) as usize, [0.0; 4]);
        // Recreate texture & view (sampler and pipeline unchanged)
        let tex_extent = wgpu::Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        };
        self.fb_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("framebuffer_texture"),
            size: tex_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.fb_view = self
            .fb_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.fb_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fb_bg"),
            layout: &self.fb_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.fb_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.fb_view),
                },
            ],
        });
    }

    pub fn update_uniforms(&mut self) { /* TODO */
    }

    pub fn render(&mut self) -> Result<()> {
        if let Some(rt) = &self.ray_tracer {
            rt.trace_frame(&mut self.framebuffer, self.config.width, self.config.height);
        }
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Upload CPU framebuffer to texture
        let bytes: &[u8] = bytemuck::cast_slice(&self.framebuffer);
        let bytes_per_row = (self.config.width as usize * std::mem::size_of::<[f32; 4]>()) as u32; // tightly packed
        let layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(self.config.height),
        };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.fb_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            layout,
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render"),
            });
        {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.fb_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
