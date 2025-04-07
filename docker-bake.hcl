group "default" {
  targets = ["image-amd64-generic", "image-arm64-graviton4"]
}

target "base" {
  dockerfile = "Dockerfile"
  context    = "."
  tags = ["ghcr.io/angt/hfendpoint-draft-llamacpp-embeddings:latest"]
}

target "image-amd64-generic" {
  inherits  = ["base"]
  platforms = ["linux/amd64"]
  args = {
    llamacpp_native           = "OFF"
    llamacpp_cpu_arm_arch     = ""
    llamacpp_backend_dl       = "ON"
    llamacpp_cpu_all_variants = "ON"
  }
  cache-from = ["type=gha,scope=amd64-generic"]
  cache-to   = ["type=gha,mode=max,scope=amd64-generic"]
}

target "image-arm64-graviton4" {
  inherits  = ["base"]
  platforms = ["linux/arm64"]
  args = {
    llamacpp_native           = "OFF"
    llamacpp_cpu_arm_arch     = "armv9-a+i8mm"
    llamacpp_backend_dl       = "OFF"
    llamacpp_cpu_all_variants = "OFF"
  }
  cache-from = ["type=gha,scope=arm64-graviton4"]
  cache-to   = ["type=gha,mode=max,scope=arm64-graviton4"]
}
