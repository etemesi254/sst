#include <tuple>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

// Links
// https://encode.su/threads/2520-MTF-There-can-be-only-one!
// https://github.com/Bulat-Ziganshin/Compression-Research/blob/master/algo_mtf/mtf_cpu_shelwien.cpp

/*
* Maximum uinque symbols we accept per byte, this is
* 256 to cover all cases
*/
#define ALPHABET_SIZE 256
/*
* Single MTF block for which all items inside this block
* are carried through MTF.
* Smaller blocks will allow blocks to fit in cache while larger
* blocks will increase compression ratio
*/
#define BLOCK_SIZE (12 * (1<<20))
/*
* Extra bytes to allocate on buffers to help with out of order reads and writed 
*/
#define SLOP_BYTES 1024

/*
* The main structure for storing MTF coefficients during iterations
*/
struct MTF {
    // Holds forward mtf sequences
    int8_t mtf_rank_f[ALPHABET_SIZE];
    // Holds inverse mtf sequence position
    uint8_t mtf_rank_b[ALPHABET_SIZE];

    /// Initialize MTF buffers.
    constexpr inline void init() {
        unsigned int i = 0;
        for (i = 0; i < ALPHABET_SIZE; i++) mtf_rank_f[i] = i - 128;
        for (i = 0; i < ALPHABET_SIZE; i++) mtf_rank_b[i] = i;
    }
};
/*
* Reverse effect of Move to Front Transform
* 
* This reverses effect of the forward transform
* 
* # Arguments
* - MTF1-4: MTF tables on that position.
* - c1-c4: MTF coded coefficients at that position
*
*  The function will modify MTF1-MTF4 structs
*
* # Returns
* Initial values before the MTF transform
*/
static std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>
reverse_four_inner(MTF &mtf1, MTF &mtf2, MTF &mtf3, MTF &mtf4, uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4) {
    unsigned int j1 = c1, j2 = c2, j3 = c3, j4 = c4;

    auto d1 = (int8_t) mtf1.mtf_rank_b[c1];
    auto d2 = (int8_t) mtf2.mtf_rank_b[c2];
    auto d3 = (int8_t) mtf3.mtf_rank_b[c3];
    auto d4 = (int8_t) mtf4.mtf_rank_b[c4];


    auto jMin = std::min({c1, c2, c3, c4});

    for (int p = jMin; p > 0; p--) {
        mtf1.mtf_rank_b[j1] = mtf1.mtf_rank_b[j1 - 1];
        mtf2.mtf_rank_b[j2] = mtf2.mtf_rank_b[j2 - 1];
        mtf3.mtf_rank_b[j3] = mtf3.mtf_rank_b[j3 - 1];
        mtf4.mtf_rank_b[j4] = mtf4.mtf_rank_b[j4 - 1];

        j1--;
        j2--;
        j3--;
        j4--;

    };

    for (; j1 > 0; j1--) mtf1.mtf_rank_b[j1] = mtf1.mtf_rank_b[j1 - 1];
    for (; j2 > 0; j2--) mtf2.mtf_rank_b[j2] = mtf2.mtf_rank_b[j2 - 1];
    for (; j3 > 0; j3--) mtf3.mtf_rank_b[j3] = mtf3.mtf_rank_b[j3 - 1];
    for (; j4 > 0; j4--) mtf4.mtf_rank_b[j4] = mtf4.mtf_rank_b[j4 - 1];

    mtf1.mtf_rank_b[0] = d1;
    mtf2.mtf_rank_b[0] = d2;
    mtf3.mtf_rank_b[0] = d3;
    mtf4.mtf_rank_b[0] = d4;

    return std::make_tuple(d1, d2, d3, d4);
}

/* 
* Perform a forward move to transform operation
* 
* 
* # Arguments
* - MTF1-4: MTF tables on that position.
* - c1-c4: Raw bytes which are to be transformed 
*
*  The function will modify MTF1-MTF4 structs
*
* # Returns
* New MTF coded transforms.
*/
static std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>
forward_four_inner(MTF &mtf1, MTF &mtf2, MTF &mtf3, MTF &mtf4, uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4) {

    unsigned int j;

    int8_t d1 = mtf1.mtf_rank_f[c1];
    int8_t d2 = mtf2.mtf_rank_f[c2];
    int8_t d3 = mtf3.mtf_rank_f[c3];
    int8_t d4 = mtf4.mtf_rank_f[c4];


    for (j = 0; j < ALPHABET_SIZE; j++) {
        mtf1.mtf_rank_f[j] -= (mtf1.mtf_rank_f[j] < d1) ? int8_t(0xFF) : int8_t(0);
        mtf2.mtf_rank_f[j] -= (mtf2.mtf_rank_f[j] < d2) ? int8_t(0xFF) : int8_t(0);
        mtf3.mtf_rank_f[j] -= (mtf3.mtf_rank_f[j] < d3) ? int8_t(0xFF) : int8_t(0);
        mtf4.mtf_rank_f[j] -= (mtf4.mtf_rank_f[j] < d4) ? int8_t(0xFF) : int8_t(0);

    }

    mtf1.mtf_rank_f[c1] = -128;
    mtf2.mtf_rank_f[c2] = -128;
    mtf3.mtf_rank_f[c3] = -128;
    mtf4.mtf_rank_f[c4] = -128;

    return std::make_tuple(d1 + 128, d2 + 128, d3 + 128, d4 + 128);
}

static void forward_four(const uint8_t *input, uint8_t *output, size_t input_length) {
    MTF m1 = MTF{};
    MTF m2 = MTF{};
    MTF m3 = MTF{};
    MTF m4 = MTF{};

    m1.init();
    m2.init();
    m3.init();
    m4.init();

    // ensure we can evenly divide this block so as not to have bytes which won't be
    // mtf-ed
    assert(input_length % 4 == 0);
    // split into four
    auto quarter = input_length / 4;

    const uint8_t *s1 = input + 0 * quarter;
    const uint8_t *s2 = input + 1 * quarter;
    const uint8_t *s3 = input + 2 * quarter;
    const uint8_t *s4 = input + 3 * quarter;

    uint8_t *o1 = output + 0 * quarter;
    uint8_t *o2 = output + 1 * quarter;
    uint8_t *o3 = output + 2 * quarter;
    uint8_t *o4 = output + 3 * quarter;

    for (int i = 0; i < quarter; i++) {

        auto resp = forward_four_inner(m1, m2, m3, m4, *s1++, *s2++, *s3++, *s4++);

        *o1++ = std::get<0>(resp);
        *o2++ = std::get<1>(resp);
        *o3++ = std::get<2>(resp);
        *o4++ = std::get<3>(resp);
    }

    // ensure we consumed all bytes
    assert(s4 - input == input_length);
}

static void reverse_four(const uint8_t *input, uint8_t *output, size_t input_length) {
    MTF m1 = MTF{};
    MTF m2 = MTF{};
    MTF m3 = MTF{};
    MTF m4 = MTF{};

    m1.init();
    m2.init();
    m3.init();
    m4.init();

    assert(input_length % 4 == 0);
    auto quarter = input_length / 4;


    // split into four
    const uint8_t *s1 = input + 0 * quarter;
    const uint8_t *s2 = input + 1 * quarter;
    const uint8_t *s3 = input + 2 * quarter;
    const uint8_t *s4 = input + 3 * quarter;

    uint8_t *o1 = output + 0 * quarter;
    uint8_t *o2 = output + 1 * quarter;
    uint8_t *o3 = output + 2 * quarter;
    uint8_t *o4 = output + 3 * quarter;

    for (int i = 0; i < quarter; i++) {

        auto resp = reverse_four_inner(m1, m2, m3, m4, *s1++, *s2++, *s3++, *s4++);

        *o1++ = std::get<0>(resp);
        *o2++ = std::get<1>(resp);
        *o3++ = std::get<2>(resp);
        *o4++ = std::get<3>(resp);
    }
}


int apply_sst(char *in_file, char *out_file) {
    int ret = 0;
    signed long fsize;
    signed long bytes_read;
    int bytes_consumed;
    size_t read_bytes = 0;


    FILE *in_fd = fopen(in_file, "r");
    FILE *out_fd = fopen(out_file, "w");

    std::vector<uint8_t> input_data;
    input_data.resize(BLOCK_SIZE + SLOP_BYTES);

    std::vector<uint8_t> output_data;
    output_data.resize(BLOCK_SIZE + SLOP_BYTES);

    if (in_fd == nullptr) {
        printf("Could not open file %s", in_file);
        ret = -1;
        goto free;
    }

    if (out_fd == nullptr) {
        printf("Could not open file %s", out_file);
        ret = -1;
        goto free;
    }
    // get file length
    fseek(in_fd, 0, SEEK_END);
    fsize = ftell(in_fd);
    fseek(in_fd, 0, SEEK_SET);

    while (fsize > 0) {
        bytes_read = fread(input_data.data(), 1, BLOCK_SIZE, in_fd);
        // make bytes read a multiple of 4.
        // the remainder will be copied as is.
        size_t block_size = (bytes_read / 4) * 4;
        read_bytes += bytes_read;

        forward_four(input_data.data(), output_data.data(), block_size);

        size_t bytes_written = fwrite(output_data.data(), 1, block_size, out_fd);

        if (bytes_written != block_size) {
            std::cout << "Mismatch in bytes written " << bytes_written << " vs " << block_size << "\n";
            goto free;
        }
        if ((bytes_read - block_size) != 0) {

            size_t diff = bytes_read - block_size;

            // some bytes were read but not processed, just write them as they are
            std::cout << "Handling extra bytes \n";
            bytes_written = fwrite(input_data.data() + block_size, 1, diff, out_fd);

            if (bytes_written != diff) {
                std::cout << "Mismatch in bytes written " << bytes_written << " vs " << diff << "\n";
                goto free;
            }
        }
        fsize -= bytes_read;
    }

free:
    if (in_fd != nullptr)
        fclose(in_fd);
    if (out_fd != nullptr)
        fclose(out_fd);
    return ret;
}


int apply_inverse_sst(char *in_file, char *out_file) {
    int ret = 0;
    signed long fsize;
    signed long bytes_read;
    int bytes_consumed;

    size_t read_bytes = 0;

    FILE *in_fd = fopen(in_file, "r");
    FILE *out_fd = fopen(out_file, "w");

    std::vector<uint8_t> input_data;
    input_data.resize(BLOCK_SIZE + SLOP_BYTES);

    std::vector<uint8_t> output_data;
    output_data.resize(BLOCK_SIZE + SLOP_BYTES);

    if (in_fd == nullptr) {
        printf("Could not open file %s", in_file);
        ret = -1;
        goto free;
    }

    if (out_fd == nullptr) {
        printf("Could not open file %s", out_file);
        ret = -1;
        goto free;
    }

    fseek(in_fd, 0, SEEK_END);
    fsize = ftell(in_fd);
    fseek(in_fd, 0, SEEK_SET);  /* same as rewind(f); */

    while (fsize > 0) {

        bytes_read = fread(input_data.data(), 1, BLOCK_SIZE, in_fd);

        // round down to nearest 4
        size_t block_size = (bytes_read / 4) * 4;
        read_bytes += bytes_read;

        reverse_four(input_data.data(), output_data.data(), block_size);

        size_t bytes_written = fwrite(output_data.data(), 1, block_size, out_fd);

        if (bytes_written != block_size) {
            std::cout << "Mismatch in bytes written " << bytes_written << " vs " << block_size;
            goto free;
        }
        if ((bytes_read - block_size) != 0) {

            size_t diff = bytes_read - block_size;
            // some bytes were read but not processed, just write them as they are
            std::cout << "Handling extra bytes \n";
            bytes_written = fwrite(input_data.data() + block_size, 1, diff, out_fd);

            if (bytes_written != diff) {
                std::cout << "Mismatch in bytes written " << bytes_written << " vs " << diff << "\n";
                goto free;
            }
        }

        fsize -= bytes_read;
    }

    free:
    if (in_fd != nullptr)
        fclose(in_fd);
    if (out_fd != nullptr)
        fclose(out_fd);
    return ret;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("[ERROR]: Too few arguments\n");
        printf("sst t <input_path> <output_path> - apply direct sst\n");
        printf("sst i <input_path> <output_path> - apply inverse sst");
        return -1;
    }
    if (strcmp(argv[1], "t") == 0) {
        // decompress routine
        // read in file
        char *in_file = argv[2];
        char *out_file = argv[3];
        apply_sst(in_file, out_file);
    } else if (strcmp(argv[1], "i") == 0) {
        // read in file
        char *in_file = argv[2];
        char *out_file = argv[3];

        apply_inverse_sst(in_file, out_file);
    }
    return 0;
}
