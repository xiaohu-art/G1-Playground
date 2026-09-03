import argparse
import subprocess

import numpy as np
import pygame

from g1_playground.policy.body_hand.depth import read_depth_preview


def parse_args():
    parser = argparse.ArgumentParser(description="View the depth observation exported by a real G1 pipeline")
    parser.add_argument("--ssh", default="unitree-wifi", help="SSH host running the real deployment pipeline")
    parser.add_argument("--port", type=int, default=9876, help="Remote localhost preview port")
    parser.add_argument("--scale", type=int, default=4, help="Integer display scale for the 128x72 observation")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-W",
        f"127.0.0.1:{args.port}",
        args.ssh,
    ]
    tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    if tunnel.stdout is None:
        raise RuntimeError("SSH did not create a depth preview stream")

    pygame.init()
    screen = None
    try:
        while True:
            sequence, image = read_depth_preview(tunnel.stdout)
            height, width = image.shape
            display_size = (width * args.scale, height * args.scale)
            if screen is None:
                screen = pygame.display.set_mode(display_size)
            rgb = np.repeat(image[:, :, None], 3, axis=2)
            surface = pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB")
            screen.blit(pygame.transform.scale(surface, display_size), (0, 0))
            pygame.display.set_caption(f"G1 depth policy observation — sequence {sequence}")
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
    except EOFError as error:
        raise RuntimeError("Depth preview is unavailable; start the real depth pipeline first") from error
    finally:
        pygame.quit()
        tunnel.terminate()
        tunnel.wait()


if __name__ == "__main__":
    run()
