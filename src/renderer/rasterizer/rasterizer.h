#pragma once

#include "resource.h"

#include <functional>
#include <iostream>
#include <linalg.h>
#include <limits>
#include <memory>


using namespace linalg::aliases;

static constexpr float DEFAULT_DEPTH = std::numeric_limits<float>::max();

namespace cg::renderer
{
	template<typename VB, typename RT>
	class rasterizer
	{
	public:
		rasterizer(){};
		~rasterizer(){};
		void set_render_target(
				std::shared_ptr<resource<RT>> in_render_target,
				std::shared_ptr<resource<float>> in_depth_buffer = nullptr);
		void clear_render_target(
				const RT& in_clear_value, const float in_depth = DEFAULT_DEPTH);

		void set_vertex_buffer(std::shared_ptr<resource<VB>> in_vertex_buffer);
		void set_index_buffer(std::shared_ptr<resource<unsigned int>> in_index_buffer);

		void set_viewport(size_t in_width, size_t in_height);

		void draw(size_t num_vertexes, size_t vertex_offset);

		std::function<std::pair<float4, VB>(float4 vertex, VB vertex_data)> vertex_shader;
		std::function<cg::color(const VB& vertex_data, const float z)> pixel_shader;

	protected:
		std::shared_ptr<cg::resource<VB>> vertex_buffer;
		std::shared_ptr<cg::resource<unsigned int>> index_buffer;
		std::shared_ptr<cg::resource<RT>> render_target;
		std::shared_ptr<cg::resource<float>> depth_buffer;

		size_t width = 1920;
		size_t height = 1080;

		int edge_function(int2 a, int2 b, int2 c);
		bool depth_test(float z, size_t x, size_t y);
	};

	template<typename VB, typename RT>
	inline void rasterizer<VB, RT>::set_render_target(
			std::shared_ptr<resource<RT>> in_render_target,
			std::shared_ptr<resource<float>> in_depth_buffer)
	{
		if (in_render_target)
			render_target = in_render_target;
		if (in_depth_buffer)
			depth_buffer = in_depth_buffer;
	}

	template<typename VB, typename RT>
	inline void rasterizer<VB, RT>::set_viewport(size_t in_width, size_t in_height)
	{
		height = in_height;
		width = in_width;
	}

	template<typename VB, typename RT>
	inline void rasterizer<VB, RT>::clear_render_target(
			const RT& in_clear_value, const float in_depth)
	{
		if (render_target)
		{
			for (size_t i = 0; i < render_target->count(); i++)
			{
				render_target->item(i) = in_clear_value;
			}
		}
		if (depth_buffer)
		{
			for (size_t i = 0; i < depth_buffer->count(); i++)
			{
				depth_buffer->item(i) = in_depth;
			}
		}
	}

	template<typename VB, typename RT>
	inline void rasterizer<VB, RT>::set_vertex_buffer(
			std::shared_ptr<resource<VB>> in_vertex_buffer)
	{
		vertex_buffer = in_vertex_buffer;
	}

	template<typename VB, typename RT>
	inline void rasterizer<VB, RT>::set_index_buffer(
			std::shared_ptr<resource<unsigned int>> in_index_buffer)
	{
		index_buffer = in_index_buffer;
	}

	template<typename VB, typename RT>
	inline void rasterizer<VB, RT>::draw(size_t num_vertexes, size_t vertex_offset)
	{
		size_t vertex_id = vertex_offset;
		while (vertex_id < vertex_offset + num_vertexes)
		{
			std::vector<VB> vertices(3);
			vertices[0] = vertex_buffer->item(
					index_buffer->item(vertex_id++));
			vertices[1] = vertex_buffer->item(
					index_buffer->item(vertex_id++));
			vertices[2] = vertex_buffer->item(
					index_buffer->item(vertex_id++));

			for (auto& vertex: vertices)
			{
				float4 coords{vertex.x, vertex.y, vertex.z, 1.f};

				auto processed_vertex = vertex_shader(coords, vertex);

				vertex.x = processed_vertex.first.x / processed_vertex.first.w;
				vertex.y = processed_vertex.first.y / processed_vertex.first.w;
				vertex.z = processed_vertex.first.z / processed_vertex.first.w;

				vertex.x = (vertex.x + 1.f) * width / 2.f;
				vertex.y = (-vertex.y + 1.f) * height / 2.f;
			}

			int2 vertex_a = float2(vertices[0].x, vertices[0].y);
			int2 vertex_b = float2(vertices[1].x, vertices[1].y);
			int2 vertex_c = float2(vertices[2].x, vertices[2].y);

			float edge = static_cast<float>(edge_function(vertex_a, vertex_b, vertex_c));

			float2 min_vertex = min(vertex_a, min(vertex_b, vertex_c));
			float2 bounding_box_begin = round(clamp(
					min_vertex,
					float2(0, 0),
					float2{static_cast<float>(width - 1),
						   static_cast<float>(height - 1)}));

			float2 max_vertex = max(vertex_a, max(vertex_b, vertex_c));
			float2 bounding_box_end = round(clamp(
					max_vertex,
					float2(0, 0),
					float2{static_cast<float>(width - 1),
						   static_cast<float>(height - 1)}));

			for (float x = bounding_box_begin.x; x <= bounding_box_end.x; x+=1.f)
			{
				for (float y = bounding_box_begin.y; y <= bounding_box_end.y; y+=1.f)
				{
					int2 point(x, y);
					int edge0 = edge_function(vertex_a, vertex_b, point);
					int edge1 = edge_function(vertex_b, vertex_c, point);
					int edge2 = edge_function(vertex_c, vertex_a, point);

					if (edge0 >= 0.f && edge1 >= 0.f && edge2 >= 0.f)
					{
						float u = edge1 / edge;
						float v = edge2 / edge;
						float w = edge0 / edge;
						float depth = u * vertices[0].z +
									  v * vertices[1].z +
									  w * vertices[2].z;

						size_t u_x = static_cast<size_t>(x);
						size_t u_y = static_cast<size_t>(y);

						if (depth_test(depth, u_x, u_y))
						{
							auto pixel_result = pixel_shader(vertices[0], depth);
							render_target->item(u_x, u_y) = RT::from_color(pixel_result);
							if (depth_buffer)
								depth_buffer->item(u_x, u_y) = depth;
						}

					}
				}
			}
		}
	}

	template<typename VB, typename RT>
	inline int
	rasterizer<VB, RT>::edge_function(int2 a, int2 b, int2 c)
	{
		return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
	}

	template<typename VB, typename RT>
	inline bool rasterizer<VB, RT>::depth_test(float z, size_t x, size_t y)
	{
		if (!depth_buffer)
		{
			return true;
		}
		return depth_buffer->item(x, y) > z;
	}

}// namespace cg::renderer