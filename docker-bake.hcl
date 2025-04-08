variable "image_name" {
  default = "hfendpoint-draft-llamacpp-embeddings"
}

group "default" {
  targets = ["amd64", "arm64-graviton4"]
}

target "base" {
  dockerfile = "Dockerfile"
  context    = "."
}

target "amd64" {
  inherits  = ["base"]
  platforms = ["linux/amd64"]
  args = {
    llamacpp_native           = "OFF"
    llamacpp_cpu_arm_arch     = ""
    llamacpp_backend_dl       = "ON"
    llamacpp_cpu_all_variants = "ON"
  }
  tags       = ["${image_name}:amd64-latest"]
  cache-from = ["type=gha,scope=${image_name}-amd64"]
  cache-to   = ["type=gha,mode=max,scope=${image_name}-amd64"]
}

target "arm64-graviton4" {
  inherits  = ["base"]
  platforms = ["linux/arm64"]
  args = {
    llamacpp_native           = "OFF"
    llamacpp_cpu_arm_arch     = "armv9-a+i8mm"
    llamacpp_backend_dl       = "OFF"
    llamacpp_cpu_all_variants = "OFF"
  }
  tags       = ["${image_name}:arm64-graviton4-latest"]
  cache-from = ["type=gha,scope=${image_name}-arm64"]
  cache-to   = ["type=gha,mode=max,scope=${image_name}-arm64"]
}
