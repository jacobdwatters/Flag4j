package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixGetRowColTests {

    CsrCMatrixOld A;
    Shape aShape;
    CNumber[] aEntries;
    int[] aRowPointers;
    int[] aColIndices;

    CooCVectorOld exp;
    int expSize;
    CNumber[] expEntries;
    int[] expIndices;

    @Test
    void getRowTests() {
        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(12, 15);
        aEntries = new CNumber[]{new CNumber(0.23602, 0.16938), new CNumber(0.96814, 0.41204),
                new CNumber(0.50047, 0.17891), new CNumber(0.78237, 0.33373), new CNumber(0.64262, 0.82639),
                new CNumber(0.09461, 0.96313), new CNumber(0.41434, 0.89052), new CNumber(0.8977, 0.7198),
                new CNumber(0.10097, 0.20405), new CNumber(0.33936, 0.09617), new CNumber(0.3967, 0.70692),
                new CNumber(0.13199, 0.02088), new CNumber(0.28428, 0.96799), new CNumber(0.28637, 0.27711),
                new CNumber(0.28657, 0.01759), new CNumber(0.75007, 0.84077), new CNumber(0.78133, 0.6493),
                new CNumber(0.61427, 0.52308)};
        aRowPointers = new int[]{0, 1, 4, 4, 7, 7, 8, 11, 12, 13, 15, 16, 18};
        aColIndices = new int[]{14, 4, 7, 9, 8, 9, 11, 11, 2, 9, 10, 12, 9, 3, 9, 11, 2, 6};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15;
        expEntries = new CNumber[]{new CNumber(0.96814, 0.41204), new CNumber(0.50047, 0.17891),
                new CNumber(0.78237, 0.33373)};
        expIndices = new int[]{4, 7, 9};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRow(1));

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(300, 15166);
        aEntries = new CNumber[]{new CNumber(0.11827, 0.35331), new CNumber(0.98714, 0.71672),
                new CNumber(0.65619, 0.40711), new CNumber(0.5775, 0.04425), new CNumber(0.03689, 0.72575),
                new CNumber(0.81466, 0.22798), new CNumber(0.15042, 0.15901), new CNumber(0.62663, 0.60531),
                new CNumber(0.42851, 0.98514), new CNumber(0.25319, 0.73223), new CNumber(0.84984, 0.49084),
                new CNumber(0.76067, 0.3437), new CNumber(0.21163, 0.111), new CNumber(0.19911, 0.51828),
                new CNumber(0.73737, 0.58554), new CNumber(0.31597, 0.61098), new CNumber(0.61981, 0.55317),
                new CNumber(0.89771, 0.08482), new CNumber(0.57691, 0.28762), new CNumber(0.0168, 0.78888),
                new CNumber(0.39332, 0.35908), new CNumber(0.4281, 0.56859), new CNumber(0.87064, 0.29464),
                new CNumber(0.53142, 0.66069), new CNumber(0.60571, 0.11065), new CNumber(0.76914, 0.87123),
                new CNumber(0.06519, 0.8469), new CNumber(0.78424, 0.43248), new CNumber(0.63513, 0.95586),
                new CNumber(0.09574, 0.56269), new CNumber(0.9692, 0.53559), new CNumber(0.54248, 0.23793),
                new CNumber(0.96551, 0.31074), new CNumber(0.15251, 0.55079), new CNumber(0.5164, 0.85289),
                new CNumber(0.69954, 0.30213), new CNumber(0.72643, 0.31503), new CNumber(0.65479, 0.25359),
                new CNumber(0.65493, 0.47835), new CNumber(0.04379, 0.10481), new CNumber(0.51433, 0.46563),
                new CNumber(0.74758, 0.79843), new CNumber(0.94717, 0.97309), new CNumber(0.42653, 0.92945),
                new CNumber(0.59163, 0.89424)};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11,
                11, 11, 12, 13, 13, 13, 13, 13, 13, 14, 14, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                17, 17, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
                29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
                31, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 36, 37, 37, 39, 39, 39, 39, 39,
                39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 45};
        aColIndices = new int[]{8644, 13316, 2475, 4742, 8613, 5339, 13064, 3856, 10265, 4381, 14646, 1492, 3871, 4174, 12022, 938,
                4211, 11533, 11423, 7119, 9759, 5107, 9, 9304, 15087, 8113, 8874, 8709, 10168, 1440, 2783, 7482, 1489, 3721, 11915,
                10588, 4684, 4947, 5088, 7959, 6683, 3493, 6879, 13401, 3782};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15166;
        expEntries = new CNumber[]{};
        expIndices = new int[]{};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRow(1));

        // --------------------- sub-case 3 ---------------------
        aShape = new Shape(300, 15166);
        aEntries = new CNumber[]{new CNumber(0.07417, 0.33776), new CNumber(0.31606, 0.60348),
                new CNumber(0.96772, 0.433), new CNumber(0.60536, 0.87768), new CNumber(0.18907, 0.01496),
                new CNumber(0.66908, 0.31327), new CNumber(0.38721, 0.07906), new CNumber(0.76406, 0.04702),
                new CNumber(0.21746, 0.97023), new CNumber(0.3385, 0.12456), new CNumber(0.72547, 0.83389),
                new CNumber(0.17455, 0.39491), new CNumber(0.79723, 0.89359), new CNumber(0.9716, 0.42416),
                new CNumber(0.75693, 0.88842), new CNumber(0.15964, 0.81973), new CNumber(0.07315, 0.58842),
                new CNumber(0.0946, 0.77006), new CNumber(0.38414, 0.02952), new CNumber(0.25713, 0.05333),
                new CNumber(0.31648, 0.45272), new CNumber(0.67666, 0.86929), new CNumber(0.2341, 0.91853),
                new CNumber(0.8324, 0.00903), new CNumber(0.5731, 0.54102), new CNumber(0.65682, 0.84017),
                new CNumber(0.86211, 0.69398), new CNumber(0.90684, 0.38901), new CNumber(0.38009, 0.90416),
                new CNumber(0.36273, 0.37741), new CNumber(0.84385, 0.49577), new CNumber(0.04411, 0.0458),
                new CNumber(0.96461, 0.01099), new CNumber(0.22804, 0.98745), new CNumber(0.75512, 0.97781),
                new CNumber(0.13956, 0.17248), new CNumber(0.0725, 0.72959), new CNumber(0.05183, 0.4561),
                new CNumber(0.81789, 0.65072), new CNumber(0.30394, 0.82701), new CNumber(0.37844, 0.48049),
                new CNumber(0.07998, 0.91007), new CNumber(0.18443, 0.2077), new CNumber(0.63667, 0.6216),
                new CNumber(0.30962, 0.56366), new CNumber(325.1, -5235.0), new CNumber(0.0, 15.0),
                new CNumber(200.25, 0.0), new CNumber(0.0025, 23.56), new CNumber(0.17583, 0.80122),
                new CNumber(0.0307, 0.41328), new CNumber(0.94518, 0.42234), new CNumber(0.91267, 0.07814),
                new CNumber(0.10082, 0.74487), new CNumber(0.08132, 0.26143), new CNumber(0.14179, 0.01315),
                new CNumber(0.17507, 0.64344), new CNumber(0.58925, 0.41227), new CNumber(0.70825, 0.84555),
                new CNumber(0.49276, 0.48589), new CNumber(0.81755, 0.39052), new CNumber(0.65007, 0.19453),
                new CNumber(0.87103, 0.70096), new CNumber(0.20915, 0.02103), new CNumber(0.38802, 0.86701),
                new CNumber(0.4608, 0.32323), new CNumber(0.73131, 0.57119), new CNumber(0.95901, 0.27634),
                new CNumber(0.35189, 0.16865), new CNumber(0.81498, 0.73806), new CNumber(0.9011, 0.97414),
                new CNumber(0.39327, 0.85636), new CNumber(0.97954, 0.32593), new CNumber(0.39307, 0.18368),
                new CNumber(0.18808, 0.82061), new CNumber(0.02764, 0.93699), new CNumber(0.12029, 0.49499),
                new CNumber(0.72848, 0.96809), new CNumber(0.50037, 0.25173), new CNumber(0.13939, 0.70156),
                new CNumber(0.49965, 0.64074), new CNumber(0.46194, 0.7433), new CNumber(0.70551, 0.67116),
                new CNumber(0.355, 0.1452), new CNumber(0.83789, 0.8059), new CNumber(0.27226, 0.25819),
                new CNumber(0.72359, 0.54591), new CNumber(0.40136, 0.72166), new CNumber(0.40853, 0.1244),
                new CNumber(0.0264, 0.4671), new CNumber(0.5316, 0.73384), new CNumber(0.04644, 0.1234),
                new CNumber(0.87873, 0.31904), new CNumber(0.17373, 0.01339), new CNumber(0.95117, 0.17136),
                new CNumber(0.97329, 0.7729), new CNumber(0.31616, 0.67813), new CNumber(0.09135, 0.94509),
                new CNumber(0.87001, 0.85635), new CNumber(0.00758, 0.17603), new CNumber(0.80255, 0.1224),
                new CNumber(0.63676, 0.36103), new CNumber(0.12557, 0.29875), new CNumber(0.73394, 0.62358),
                new CNumber(0.09183, 0.55434), new CNumber(0.68072, 0.92796), new CNumber(0.2264, 0.76327),
                new CNumber(0.18836, 0.7597), new CNumber(0.47364, 0.43289), new CNumber(0.90175, 0.5762),
                new CNumber(0.08266, 0.04132), new CNumber(0.19222, 0.83772), new CNumber(0.92886, 0.66696),
                new CNumber(0.9272, 0.32297), new CNumber(0.65714, 0.11511), new CNumber(0.9301, 0.92064),
                new CNumber(0.05519, 0.49315), new CNumber(0.70471, 0.67155), new CNumber(0.87287, 0.03979),
                new CNumber(0.42453, 0.0771), new CNumber(0.75881, 0.55626), new CNumber(0.44765, 0.06961),
                new CNumber(0.31151, 0.21722), new CNumber(0.97534, 0.21786), new CNumber(0.32177, 0.54134),
                new CNumber(0.54665, 0.26363), new CNumber(0.00402, 0.32476), new CNumber(0.50158, 0.65639),
                new CNumber(0.13285, 0.17358), new CNumber(0.40227, 0.34524), new CNumber(0.62518, 0.15134),
                new CNumber(0.58675, 0.4635), new CNumber(0.99354, 0.44441), new CNumber(0.20614, 0.06949),
                new CNumber(0.78794, 0.81834), new CNumber(0.41006, 0.24625), new CNumber(0.3653, 0.78255),
                new CNumber(0.05381, 0.60392), new CNumber(0.21342, 0.37019), new CNumber(0.88689, 0.36104),
                new CNumber(0.18875, 0.01533), new CNumber(0.03235, 0.63991), new CNumber(0.80908, 0.72261),
                new CNumber(0.46771, 0.29074), new CNumber(0.16719, 0.98106), new CNumber(0.27523, 0.6026),
                new CNumber(0.58742, 0.38022), new CNumber(0.90384, 0.09489), new CNumber(0.26217, 0.60955),
                new CNumber(0.63225, 0.56871), new CNumber(0.35017, 0.65807), new CNumber(0.49884, 0.02367),
                new CNumber(0.89073, 0.46354), new CNumber(0.50107, 0.20768), new CNumber(0.78063, 0.64339),
                new CNumber(0.46752, 0.24795), new CNumber(0.7781, 0.74621), new CNumber(0.67593, 0.08227),
                new CNumber(0.04364, 0.92822), new CNumber(0.0748, 0.99679), new CNumber(0.82186, 0.64914),
                new CNumber(0.8053, 0.88578), new CNumber(0.36582, 0.1991), new CNumber(0.68069, 0.14296),
                new CNumber(0.31057, 0.72957), new CNumber(0.22528, 0.50156), new CNumber(0.74779, 0.46033),
                new CNumber(0.29114, 0.26132), new CNumber(0.74951, 0.18053), new CNumber(0.15152, 0.87689),
                new CNumber(0.88389, 0.62622), new CNumber(0.19534, 0.29521), new CNumber(0.44199, 0.58656),
                new CNumber(0.58043, 0.99993), new CNumber(0.07742, 0.36248), new CNumber(0.50984, 0.34393),
                new CNumber(0.17955, 0.78658), new CNumber(0.75337, 0.23502), new CNumber(0.58356, 0.5011),
                new CNumber(0.69879, 0.24202), new CNumber(0.60726, 0.31803), new CNumber(0.3807, 0.78102),
                new CNumber(0.83648, 0.69741), new CNumber(0.94284, 0.46428), new CNumber(0.99928, 0.24795),
                new CNumber(0.25366, 0.95709), new CNumber(0.54861, 0.93149), new CNumber(0.33885, 0.44512),
                new CNumber(0.90328, 0.38962), new CNumber(0.54871, 0.9487), new CNumber(0.2579, 0.33187),
                new CNumber(0.979, 0.19838), new CNumber(0.7881, 0.93007), new CNumber(0.98205, 0.20834),
                new CNumber(0.81939, 0.38326), new CNumber(0.43123, 0.13852), new CNumber(0.53469, 0.31964),
                new CNumber(0.66996, 0.63725), new CNumber(0.25209, 0.34233), new CNumber(0.57683, 0.16357),
                new CNumber(0.45884, 0.73653), new CNumber(0.19001, 0.41945), new CNumber(0.1327, 0.80527),
                new CNumber(0.03333, 0.85803), new CNumber(0.97391, 0.02655), new CNumber(0.11362, 0.47215),
                new CNumber(0.20636, 0.88407), new CNumber(0.06771, 0.98198), new CNumber(0.5518, 0.35977),
                new CNumber(0.91907, 0.0687), new CNumber(0.23925, 0.05597), new CNumber(0.81963, 0.32057),
                new CNumber(0.9985, 0.39285), new CNumber(0.12062, 0.45297), new CNumber(0.8906, 0.86926),
                new CNumber(0.22809, 0.84537), new CNumber(0.62687, 0.1817), new CNumber(0.44058, 0.53357),
                new CNumber(0.37758, 0.10782), new CNumber(0.08602, 0.54584), new CNumber(0.21486, 0.49503),
                new CNumber(0.48271, 0.67872), new CNumber(0.80135, 0.44841), new CNumber(0.27364, 0.75899),
                new CNumber(0.35212, 0.11285), new CNumber(0.27263, 0.03466), new CNumber(0.88125, 0.78596),
                new CNumber(0.667, 0.84789), new CNumber(0.00733, 0.23049), new CNumber(0.45477, 0.28087),
                new CNumber(0.09393, 0.98723)};
        aRowPointers = new int[]{0, 2, 2, 3, 3, 4, 4, 4, 6, 6, 6, 9, 9, 12, 13, 14, 14, 14, 16, 16, 17, 17, 17, 18, 19, 21, 21, 21,
                21, 21, 23, 23, 23, 24, 25, 27, 28, 29, 30, 32, 33, 35, 35, 35, 35, 35, 35, 36, 38, 39, 40, 40, 40, 41, 41, 41, 41,
                41, 41, 41, 41, 43, 44, 44, 45, 45, 45, 50, 54, 54, 55, 56, 57, 57, 58, 59, 60, 61, 61, 62, 62, 62, 65, 65, 65, 65,
                66, 66, 66, 66, 66, 66, 66, 67, 67, 68, 68, 68, 70, 71, 71, 74, 76, 76, 76, 76, 77, 78, 79, 80, 81, 82, 84, 84, 85,
                86, 86, 86, 87, 90, 90, 90, 91, 91, 91, 91, 93, 93, 94, 95, 95, 96, 97, 99, 99, 99, 100, 100, 100, 102, 103, 104, 104,
                105, 106, 106, 106, 108, 108, 109, 109, 110, 110, 110, 110, 110, 110, 112, 112, 114, 115, 116, 117, 118, 119, 119, 119,
                121, 122, 122, 122, 122, 122, 123, 123, 124, 125, 125, 127, 128, 129, 130, 131, 131, 131, 131, 131, 131, 131, 131, 132,
                132, 132, 133, 135, 135, 136, 138, 138, 138, 139, 141, 142, 143, 143, 146, 149, 149, 151, 152, 152, 153, 153, 154, 154,
                157, 158, 158, 159, 160, 164, 165, 166, 168, 169, 171, 171, 173, 173, 175, 175, 175, 176, 176, 176, 176, 176, 176, 178,
                178, 179, 180, 180, 181, 182, 182, 182, 182, 182, 185, 187, 188, 191, 191, 193, 193, 195, 198, 200, 201, 202, 204, 205,
                206, 206, 206, 206, 206, 207, 207, 207, 207, 207, 207, 207, 209, 209, 211, 214, 215, 216, 216, 218, 219, 220, 220, 221,
                223, 223, 223, 225, 225, 225, 226, 226, 228, 230, 230, 230, 230, 231};
        aColIndices = new int[]{11308, 14644, 3093, 8620, 1616, 8706, 3974, 10204, 11162, 3423, 8855, 10055, 5002, 4350, 4639, 14213,
                7561, 8984, 13917, 11119, 14597, 5049, 11943, 891, 9757, 11131, 13557, 9352, 1393, 9774, 9689, 13741, 266, 3308, 11418,
                3848, 6860, 9572, 9517, 10898, 7181, 3654, 14014, 11088, 15150, 14, 152, 256, 299, 2051, 4631, 8720, 9872, 10621,
                12123, 4093, 5528, 5623, 5555, 13158, 498, 7874, 7759, 11179, 15136, 1182, 2174, 7127, 1473, 3703, 6918, 5471, 10667,
                13453, 4719, 5669, 5848, 9375, 1552, 3924, 5768, 4041, 933, 2312, 11382, 10648, 201, 7395, 11920, 13169, 3501, 7369,
                7628, 6518, 6723, 2468, 7912, 4276, 13528, 7293, 3724, 4350, 5104, 13582, 8593, 4325, 1354, 10690, 8653, 9559, 9159,
                11269, 352, 2824, 706, 4536, 5060, 860, 11423, 7442, 11061, 3993, 12585, 8023, 4048, 5590, 9877, 12499, 9861, 2130,
                3196, 6442, 2637, 2628, 10739, 13330, 12679, 13951, 12340, 4474, 6147, 10623, 4005, 3746, 5426, 14336, 4897, 9224,
                10509, 6763, 7543, 5665, 11728, 4805, 1337, 1405, 4177, 103, 52, 3227, 7151, 11508, 14054, 15148, 14836, 14235, 5331,
                9556, 3232, 2788, 4389, 5476, 14200, 7503, 8578, 8262, 1446, 13548, 7602, 1858, 14746, 12374, 4150, 6407, 8700, 4406,
                13511, 7287, 86, 1028, 13812, 3358, 13245, 9521, 11240, 1192, 6935, 13412, 10208, 14909, 8296, 13444, 4531, 9687,
                10261, 8491, 14865, 1128, 6940, 41, 7431, 5395, 10194, 11774, 4126, 2777, 1447, 3493, 8081, 10003, 14514, 1622, 13315,
                11606, 12628, 9514, 889, 10978, 4847, 12431, 10487};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15166;
        expEntries = new CNumber[]{new CNumber(325.1, -5235.0), new CNumber(0.0, 15.0), new CNumber(200.25, 0.0),
                new CNumber(0.0025, 23.56), new CNumber(0.17583, 0.80122)};
        expIndices = new int[]{14, 152, 256, 299, 2051};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRow(66));

        // --------------------- sub-case 4 ---------------------
        A = new CsrCMatrixOld(1000, 15235);
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(-1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(1001));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(-4));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRow(20015));
    }


    @Test
    void getRowAfterTests() {
        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(12, 15);
        aEntries = new CNumber[]{new CNumber(0.23602, 0.16938), new CNumber(0.96814, 0.41204),
                new CNumber(0.50047, 0.17891), new CNumber(0.78237, 0.33373), new CNumber(0.64262, 0.82639),
                new CNumber(0.09461, 0.96313), new CNumber(0.41434, 0.89052), new CNumber(0.8977, 0.7198),
                new CNumber(0.10097, 0.20405), new CNumber(0.33936, 0.09617), new CNumber(0.3967, 0.70692),
                new CNumber(0.13199, 0.02088), new CNumber(0.28428, 0.96799), new CNumber(0.28637, 0.27711),
                new CNumber(0.28657, 0.01759), new CNumber(0.75007, 0.84077), new CNumber(0.78133, 0.6493),
                new CNumber(0.61427, 0.52308)};
        aRowPointers = new int[]{0, 1, 4, 4, 7, 7, 8, 11, 12, 13, 15, 16, 18};
        aColIndices = new int[]{14, 4, 7, 9, 8, 9, 11, 11, 2, 9, 10, 12, 9, 3, 9, 11, 2, 6};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 10;
        expEntries = new CNumber[]{new CNumber(0.50047, 0.17891),
                new CNumber(0.78237, 0.33373)};
        expIndices = new int[]{7-5, 9-5};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRowAfter(5, 1));

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(300, 15166);
        aEntries = new CNumber[]{new CNumber(0.11827, 0.35331), new CNumber(0.98714, 0.71672),
                new CNumber(0.65619, 0.40711), new CNumber(0.5775, 0.04425), new CNumber(0.03689, 0.72575),
                new CNumber(0.81466, 0.22798), new CNumber(0.15042, 0.15901), new CNumber(0.62663, 0.60531),
                new CNumber(0.42851, 0.98514), new CNumber(0.25319, 0.73223), new CNumber(0.84984, 0.49084),
                new CNumber(0.76067, 0.3437), new CNumber(0.21163, 0.111), new CNumber(0.19911, 0.51828),
                new CNumber(0.73737, 0.58554), new CNumber(0.31597, 0.61098), new CNumber(0.61981, 0.55317),
                new CNumber(0.89771, 0.08482), new CNumber(0.57691, 0.28762), new CNumber(0.0168, 0.78888),
                new CNumber(0.39332, 0.35908), new CNumber(0.4281, 0.56859), new CNumber(0.87064, 0.29464),
                new CNumber(0.53142, 0.66069), new CNumber(0.60571, 0.11065), new CNumber(0.76914, 0.87123),
                new CNumber(0.06519, 0.8469), new CNumber(0.78424, 0.43248), new CNumber(0.63513, 0.95586),
                new CNumber(0.09574, 0.56269), new CNumber(0.9692, 0.53559), new CNumber(0.54248, 0.23793),
                new CNumber(0.96551, 0.31074), new CNumber(0.15251, 0.55079), new CNumber(0.5164, 0.85289),
                new CNumber(0.69954, 0.30213), new CNumber(0.72643, 0.31503), new CNumber(0.65479, 0.25359),
                new CNumber(0.65493, 0.47835), new CNumber(0.04379, 0.10481), new CNumber(0.51433, 0.46563),
                new CNumber(0.74758, 0.79843), new CNumber(0.94717, 0.97309), new CNumber(0.42653, 0.92945),
                new CNumber(0.59163, 0.89424)};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11,
                11, 11, 12, 13, 13, 13, 13, 13, 13, 14, 14, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
                17, 17, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
                29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
                31, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 36, 37, 37, 39, 39, 39, 39, 39,
                39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 45};
        aColIndices = new int[]{8644, 13316, 2475, 4742, 8613, 5339, 13064, 3856, 10265, 4381, 14646, 1492, 3871, 4174, 12022, 938,
                4211, 11533, 11423, 7119, 9759, 5107, 9, 9304, 15087, 8113, 8874, 8709, 10168, 1440, 2783, 7482, 1489, 3721, 11915,
                10588, 4684, 4947, 5088, 7959, 6683, 3493, 6879, 13401, 3782};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15166-67;
        expEntries = new CNumber[]{};
        expIndices = new int[]{};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRowAfter(67, 1));

        // --------------------- sub-case 3 ---------------------
        aShape = new Shape(300, 15166);
        aEntries = new CNumber[]{new CNumber(0.07417, 0.33776), new CNumber(0.31606, 0.60348),
                new CNumber(0.96772, 0.433), new CNumber(0.60536, 0.87768), new CNumber(0.18907, 0.01496),
                new CNumber(0.66908, 0.31327), new CNumber(0.38721, 0.07906), new CNumber(0.76406, 0.04702),
                new CNumber(0.21746, 0.97023), new CNumber(0.3385, 0.12456), new CNumber(0.72547, 0.83389),
                new CNumber(0.17455, 0.39491), new CNumber(0.79723, 0.89359), new CNumber(0.9716, 0.42416),
                new CNumber(0.75693, 0.88842), new CNumber(0.15964, 0.81973), new CNumber(0.07315, 0.58842),
                new CNumber(0.0946, 0.77006), new CNumber(0.38414, 0.02952), new CNumber(0.25713, 0.05333),
                new CNumber(0.31648, 0.45272), new CNumber(0.67666, 0.86929), new CNumber(0.2341, 0.91853),
                new CNumber(0.8324, 0.00903), new CNumber(0.5731, 0.54102), new CNumber(0.65682, 0.84017),
                new CNumber(0.86211, 0.69398), new CNumber(0.90684, 0.38901), new CNumber(0.38009, 0.90416),
                new CNumber(0.36273, 0.37741), new CNumber(0.84385, 0.49577), new CNumber(0.04411, 0.0458),
                new CNumber(0.96461, 0.01099), new CNumber(0.22804, 0.98745), new CNumber(0.75512, 0.97781),
                new CNumber(0.13956, 0.17248), new CNumber(0.0725, 0.72959), new CNumber(0.05183, 0.4561),
                new CNumber(0.81789, 0.65072), new CNumber(0.30394, 0.82701), new CNumber(0.37844, 0.48049),
                new CNumber(0.07998, 0.91007), new CNumber(0.18443, 0.2077), new CNumber(0.63667, 0.6216),
                new CNumber(0.30962, 0.56366), new CNumber(325.1, -5235.0), new CNumber(0.0, 15.0),
                new CNumber(200.25, 0.0), new CNumber(0.0025, 23.56), new CNumber(0.17583, 0.80122),
                new CNumber(0.0307, 0.41328), new CNumber(0.94518, 0.42234), new CNumber(0.91267, 0.07814),
                new CNumber(0.10082, 0.74487), new CNumber(0.08132, 0.26143), new CNumber(0.14179, 0.01315),
                new CNumber(0.17507, 0.64344), new CNumber(0.58925, 0.41227), new CNumber(0.70825, 0.84555),
                new CNumber(0.49276, 0.48589), new CNumber(0.81755, 0.39052), new CNumber(0.65007, 0.19453),
                new CNumber(0.87103, 0.70096), new CNumber(0.20915, 0.02103), new CNumber(0.38802, 0.86701),
                new CNumber(0.4608, 0.32323), new CNumber(0.73131, 0.57119), new CNumber(0.95901, 0.27634),
                new CNumber(0.35189, 0.16865), new CNumber(0.81498, 0.73806), new CNumber(0.9011, 0.97414),
                new CNumber(0.39327, 0.85636), new CNumber(0.97954, 0.32593), new CNumber(0.39307, 0.18368),
                new CNumber(0.18808, 0.82061), new CNumber(0.02764, 0.93699), new CNumber(0.12029, 0.49499),
                new CNumber(0.72848, 0.96809), new CNumber(0.50037, 0.25173), new CNumber(0.13939, 0.70156),
                new CNumber(0.49965, 0.64074), new CNumber(0.46194, 0.7433), new CNumber(0.70551, 0.67116),
                new CNumber(0.355, 0.1452), new CNumber(0.83789, 0.8059), new CNumber(0.27226, 0.25819),
                new CNumber(0.72359, 0.54591), new CNumber(0.40136, 0.72166), new CNumber(0.40853, 0.1244),
                new CNumber(0.0264, 0.4671), new CNumber(0.5316, 0.73384), new CNumber(0.04644, 0.1234),
                new CNumber(0.87873, 0.31904), new CNumber(0.17373, 0.01339), new CNumber(0.95117, 0.17136),
                new CNumber(0.97329, 0.7729), new CNumber(0.31616, 0.67813), new CNumber(0.09135, 0.94509),
                new CNumber(0.87001, 0.85635), new CNumber(0.00758, 0.17603), new CNumber(0.80255, 0.1224),
                new CNumber(0.63676, 0.36103), new CNumber(0.12557, 0.29875), new CNumber(0.73394, 0.62358),
                new CNumber(0.09183, 0.55434), new CNumber(0.68072, 0.92796), new CNumber(0.2264, 0.76327),
                new CNumber(0.18836, 0.7597), new CNumber(0.47364, 0.43289), new CNumber(0.90175, 0.5762),
                new CNumber(0.08266, 0.04132), new CNumber(0.19222, 0.83772), new CNumber(0.92886, 0.66696),
                new CNumber(0.9272, 0.32297), new CNumber(0.65714, 0.11511), new CNumber(0.9301, 0.92064),
                new CNumber(0.05519, 0.49315), new CNumber(0.70471, 0.67155), new CNumber(0.87287, 0.03979),
                new CNumber(0.42453, 0.0771), new CNumber(0.75881, 0.55626), new CNumber(0.44765, 0.06961),
                new CNumber(0.31151, 0.21722), new CNumber(0.97534, 0.21786), new CNumber(0.32177, 0.54134),
                new CNumber(0.54665, 0.26363), new CNumber(0.00402, 0.32476), new CNumber(0.50158, 0.65639),
                new CNumber(0.13285, 0.17358), new CNumber(0.40227, 0.34524), new CNumber(0.62518, 0.15134),
                new CNumber(0.58675, 0.4635), new CNumber(0.99354, 0.44441), new CNumber(0.20614, 0.06949),
                new CNumber(0.78794, 0.81834), new CNumber(0.41006, 0.24625), new CNumber(0.3653, 0.78255),
                new CNumber(0.05381, 0.60392), new CNumber(0.21342, 0.37019), new CNumber(0.88689, 0.36104),
                new CNumber(0.18875, 0.01533), new CNumber(0.03235, 0.63991), new CNumber(0.80908, 0.72261),
                new CNumber(0.46771, 0.29074), new CNumber(0.16719, 0.98106), new CNumber(0.27523, 0.6026),
                new CNumber(0.58742, 0.38022), new CNumber(0.90384, 0.09489), new CNumber(0.26217, 0.60955),
                new CNumber(0.63225, 0.56871), new CNumber(0.35017, 0.65807), new CNumber(0.49884, 0.02367),
                new CNumber(0.89073, 0.46354), new CNumber(0.50107, 0.20768), new CNumber(0.78063, 0.64339),
                new CNumber(0.46752, 0.24795), new CNumber(0.7781, 0.74621), new CNumber(0.67593, 0.08227),
                new CNumber(0.04364, 0.92822), new CNumber(0.0748, 0.99679), new CNumber(0.82186, 0.64914),
                new CNumber(0.8053, 0.88578), new CNumber(0.36582, 0.1991), new CNumber(0.68069, 0.14296),
                new CNumber(0.31057, 0.72957), new CNumber(0.22528, 0.50156), new CNumber(0.74779, 0.46033),
                new CNumber(0.29114, 0.26132), new CNumber(0.74951, 0.18053), new CNumber(0.15152, 0.87689),
                new CNumber(0.88389, 0.62622), new CNumber(0.19534, 0.29521), new CNumber(0.44199, 0.58656),
                new CNumber(0.58043, 0.99993), new CNumber(0.07742, 0.36248), new CNumber(0.50984, 0.34393),
                new CNumber(0.17955, 0.78658), new CNumber(0.75337, 0.23502), new CNumber(0.58356, 0.5011),
                new CNumber(0.69879, 0.24202), new CNumber(0.60726, 0.31803), new CNumber(0.3807, 0.78102),
                new CNumber(0.83648, 0.69741), new CNumber(0.94284, 0.46428), new CNumber(0.99928, 0.24795),
                new CNumber(0.25366, 0.95709), new CNumber(0.54861, 0.93149), new CNumber(0.33885, 0.44512),
                new CNumber(0.90328, 0.38962), new CNumber(0.54871, 0.9487), new CNumber(0.2579, 0.33187),
                new CNumber(0.979, 0.19838), new CNumber(0.7881, 0.93007), new CNumber(0.98205, 0.20834),
                new CNumber(0.81939, 0.38326), new CNumber(0.43123, 0.13852), new CNumber(0.53469, 0.31964),
                new CNumber(0.66996, 0.63725), new CNumber(0.25209, 0.34233), new CNumber(0.57683, 0.16357),
                new CNumber(0.45884, 0.73653), new CNumber(0.19001, 0.41945), new CNumber(0.1327, 0.80527),
                new CNumber(0.03333, 0.85803), new CNumber(0.97391, 0.02655), new CNumber(0.11362, 0.47215),
                new CNumber(0.20636, 0.88407), new CNumber(0.06771, 0.98198), new CNumber(0.5518, 0.35977),
                new CNumber(0.91907, 0.0687), new CNumber(0.23925, 0.05597), new CNumber(0.81963, 0.32057),
                new CNumber(0.9985, 0.39285), new CNumber(0.12062, 0.45297), new CNumber(0.8906, 0.86926),
                new CNumber(0.22809, 0.84537), new CNumber(0.62687, 0.1817), new CNumber(0.44058, 0.53357),
                new CNumber(0.37758, 0.10782), new CNumber(0.08602, 0.54584), new CNumber(0.21486, 0.49503),
                new CNumber(0.48271, 0.67872), new CNumber(0.80135, 0.44841), new CNumber(0.27364, 0.75899),
                new CNumber(0.35212, 0.11285), new CNumber(0.27263, 0.03466), new CNumber(0.88125, 0.78596),
                new CNumber(0.667, 0.84789), new CNumber(0.00733, 0.23049), new CNumber(0.45477, 0.28087),
                new CNumber(0.09393, 0.98723)};
        aRowPointers = new int[]{0, 2, 2, 3, 3, 4, 4, 4, 6, 6, 6, 9, 9, 12, 13, 14, 14, 14, 16, 16, 17, 17, 17, 18, 19, 21, 21, 21,
                21, 21, 23, 23, 23, 24, 25, 27, 28, 29, 30, 32, 33, 35, 35, 35, 35, 35, 35, 36, 38, 39, 40, 40, 40, 41, 41, 41, 41,
                41, 41, 41, 41, 43, 44, 44, 45, 45, 45, 50, 54, 54, 55, 56, 57, 57, 58, 59, 60, 61, 61, 62, 62, 62, 65, 65, 65, 65,
                66, 66, 66, 66, 66, 66, 66, 67, 67, 68, 68, 68, 70, 71, 71, 74, 76, 76, 76, 76, 77, 78, 79, 80, 81, 82, 84, 84, 85,
                86, 86, 86, 87, 90, 90, 90, 91, 91, 91, 91, 93, 93, 94, 95, 95, 96, 97, 99, 99, 99, 100, 100, 100, 102, 103, 104, 104,
                105, 106, 106, 106, 108, 108, 109, 109, 110, 110, 110, 110, 110, 110, 112, 112, 114, 115, 116, 117, 118, 119, 119, 119,
                121, 122, 122, 122, 122, 122, 123, 123, 124, 125, 125, 127, 128, 129, 130, 131, 131, 131, 131, 131, 131, 131, 131, 132,
                132, 132, 133, 135, 135, 136, 138, 138, 138, 139, 141, 142, 143, 143, 146, 149, 149, 151, 152, 152, 153, 153, 154, 154,
                157, 158, 158, 159, 160, 164, 165, 166, 168, 169, 171, 171, 173, 173, 175, 175, 175, 176, 176, 176, 176, 176, 176, 178,
                178, 179, 180, 180, 181, 182, 182, 182, 182, 182, 185, 187, 188, 191, 191, 193, 193, 195, 198, 200, 201, 202, 204, 205,
                206, 206, 206, 206, 206, 207, 207, 207, 207, 207, 207, 207, 209, 209, 211, 214, 215, 216, 216, 218, 219, 220, 220, 221,
                223, 223, 223, 225, 225, 225, 226, 226, 228, 230, 230, 230, 230, 231};
        aColIndices = new int[]{11308, 14644, 3093, 8620, 1616, 8706, 3974, 10204, 11162, 3423, 8855, 10055, 5002, 4350, 4639, 14213,
                7561, 8984, 13917, 11119, 14597, 5049, 11943, 891, 9757, 11131, 13557, 9352, 1393, 9774, 9689, 13741, 266, 3308, 11418,
                3848, 6860, 9572, 9517, 10898, 7181, 3654, 14014, 11088, 15150, 14, 152, 256, 299, 2051, 4631, 8720, 9872, 10621,
                12123, 4093, 5528, 5623, 5555, 13158, 498, 7874, 7759, 11179, 15136, 1182, 2174, 7127, 1473, 3703, 6918, 5471, 10667,
                13453, 4719, 5669, 5848, 9375, 1552, 3924, 5768, 4041, 933, 2312, 11382, 10648, 201, 7395, 11920, 13169, 3501, 7369,
                7628, 6518, 6723, 2468, 7912, 4276, 13528, 7293, 3724, 4350, 5104, 13582, 8593, 4325, 1354, 10690, 8653, 9559, 9159,
                11269, 352, 2824, 706, 4536, 5060, 860, 11423, 7442, 11061, 3993, 12585, 8023, 4048, 5590, 9877, 12499, 9861, 2130,
                3196, 6442, 2637, 2628, 10739, 13330, 12679, 13951, 12340, 4474, 6147, 10623, 4005, 3746, 5426, 14336, 4897, 9224,
                10509, 6763, 7543, 5665, 11728, 4805, 1337, 1405, 4177, 103, 52, 3227, 7151, 11508, 14054, 15148, 14836, 14235, 5331,
                9556, 3232, 2788, 4389, 5476, 14200, 7503, 8578, 8262, 1446, 13548, 7602, 1858, 14746, 12374, 4150, 6407, 8700, 4406,
                13511, 7287, 86, 1028, 13812, 3358, 13245, 9521, 11240, 1192, 6935, 13412, 10208, 14909, 8296, 13444, 4531, 9687,
                10261, 8491, 14865, 1128, 6940, 41, 7431, 5395, 10194, 11774, 4126, 2777, 1447, 3493, 8081, 10003, 14514, 1622, 13315,
                11606, 12628, 9514, 889, 10978, 4847, 12431, 10487};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15166-180;
        expEntries = new CNumber[]{new CNumber(200.25, 0.0), new CNumber(0.0025, 23.56), new CNumber(0.17583, 0.80122)};
        expIndices = new int[]{256-180, 299-180, 2051-180};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getRowAfter(180, 66));

        // --------------------- sub-case 4 ---------------------
        A = new CsrCMatrixOld(1000, 15235);
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(34, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(2, 1001));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(-1, -4));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getRowAfter(9026910, 20015));
    }


    @Test
    void getColTests() {
        // --------------------- sub-case 1 ---------------------
        aShape = new Shape(15, 15);
        aEntries = new CNumber[]{new CNumber(0.55665, 0.29606), new CNumber(0.21818, 0.9712), new CNumber(0.65279, 0.31613), new CNumber(0.86714, 0.28674), new CNumber(0.50446, 0.76213), new CNumber(0.8378, 0.74914), new CNumber(0.32969, 0.9887), new CNumber(0.72675, 0.56593), new CNumber(0.74808, 0.42695), new CNumber(0.34951, 0.81078), new CNumber(0.42019, 0.15711), new CNumber(0.5746, 0.56468), new CNumber(0.7098, 0.22421), new CNumber(0.16338, 0.84483), new CNumber(0.7074, 0.46212), new CNumber(0.06522, 0.53442), new CNumber(0.24251, 0.22357), new CNumber(0.1591, 0.83203), new CNumber(0.68286, 0.08171), new CNumber(0.49218, 0.00795), new CNumber(0.77747, 0.88406), new CNumber(0.21185, 0.52285), new CNumber(0.18112, 0.7577), new CNumber(0.92702, 0.05993), new CNumber(0.33354, 0.22838), new CNumber(0.28677, 0.89053), new CNumber(0.67063, 0.97222), new CNumber(0.45945, 0.2992), new CNumber(0.34819, 0.80283), new CNumber(0.57362, 0.37478), new CNumber(0.88969, 0.69861), new CNumber(0.47338, 0.92047), new CNumber(0.59016, 0.75278), new CNumber(0.59822, 0.25406), new CNumber(0.88912, 0.33943), new CNumber(0.80134, 0.68503), new CNumber(0.47818, 0.03871), new CNumber(0.59592, 0.04876), new CNumber(0.49683, 0.62696), new CNumber(0.82016, 0.33223), new CNumber(0.22196, 0.16496), new CNumber(0.99328, 0.70379), new CNumber(0.09211, 0.98179), new CNumber(0.7068, 0.32362), new CNumber(0.02528, 0.23349), new CNumber(0.05837, 0.01675), new CNumber(0.50625, 0.24137), new CNumber(0.84099, 0.82527), new CNumber(0.52687, 0.67906), new CNumber(0.27742, 0.53553), new CNumber(0.40925, 0.95997), new CNumber(0.43899, 0.37746), new CNumber(0.8819, 0.30187), new CNumber(0.87222, 0.76276), new CNumber(0.68874, 0.00434), new CNumber(0.28473, 0.60034)};
        aRowPointers = new int[]{0, 5, 8, 9, 14, 16, 18, 22, 26, 32, 36, 41, 46, 49, 51, 56};
        aColIndices = new int[]{5, 7, 8, 9, 12, 6, 9, 13, 14, 2, 7, 10, 11, 12, 8, 9, 3, 4, 5, 11, 12, 14, 0, 1, 5, 11, 1, 4, 6, 8, 9, 12, 4, 5, 6, 8, 0, 8, 11, 12, 13, 2, 3, 6, 9, 13, 7, 11, 12, 3, 13, 1, 2, 4, 9, 14};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 15;
        expEntries = new CNumber[]{new CNumber(0.24251, 0.22357), new CNumber(0.09211, 0.98179), new CNumber(0.27742, 0.53553)};
        expIndices = new int[]{5, 11, 13};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getCol(3));

        // --------------------- sub-case 2 ---------------------
        aShape = new Shape(130, 156);
        aEntries = new CNumber[]{new CNumber(0.99107, 0.03301), new CNumber(0.29454, 0.07323), new CNumber(0.62045, 0.18456), new CNumber(0.93435, 0.60405), new CNumber(0.7067, 0.43003), new CNumber(15.6, -9.3), new CNumber(0.83198, 0.09194), new CNumber(0.65607, 0.04688), new CNumber(0.98788, 0.58681), new CNumber(0.36453, 0.77038), new CNumber(0.44178, 0.48798), new CNumber(0.59911, 0.50006), new CNumber(0.80486, 0.42829), new CNumber(0.9365, 0.85784), new CNumber(0.54784, 0.89768), new CNumber(0.78264, 0.7762), new CNumber(0.66632, 0.04367), new CNumber(325.6, -0.0015), new CNumber(0.59403, 0.56382), new CNumber(0.62668, 0.24169), new CNumber(0.92688, 0.03511), new CNumber(0.8457, 0.13989), new CNumber(0.1006, 0.97042), new CNumber(0.02962, 0.54478), new CNumber(0.18817, 0.34006), new CNumber(0.9392, 0.5656), new CNumber(0.65649, 0.42331), new CNumber(0.99858, 0.3669), new CNumber(0.94221, 0.31404), new CNumber(0.05941, 0.33145), new CNumber(0.85704, 0.0076), new CNumber(0.94471, 0.01803), new CNumber(0.84173, 0.84047), new CNumber(0.56608, 0.57495), new CNumber(0.74177, 0.53429), new CNumber(0.29672, 0.57347), new CNumber(0.2044, 0.73664), new CNumber(0.7223, 0.02033), new CNumber(0.43509, 0.80003), new CNumber(0.38461, 0.87307), new CNumber(0.84572, 0.61301), new CNumber(0.33431, 0.82291), new CNumber(0.44181, 0.60089), new CNumber(0.45548, 0.51822), new CNumber(0.87477, 0.80462), new CNumber(0.59899, 0.586), new CNumber(0.74317, 0.3269), new CNumber(0.1467, 0.18467), new CNumber(0.44001, 0.43861), new CNumber(0.91416, 0.35709), new CNumber(0.76048, 0.56933), new CNumber(0.59212, 0.89603), new CNumber(0.88027, 0.62336), new CNumber(0.00661, 0.70341), new CNumber(0.66716, 0.86243), new CNumber(0.22113, 0.70094), new CNumber(0.52961, 0.16805), new CNumber(0.5109, 0.21496), new CNumber(0.0108, 0.83879), new CNumber(0.73692, 0.35065), new CNumber(0.75467, 0.67288), new CNumber(0.28416, 0.5289), new CNumber(0.56294, 0.46089), new CNumber(0.30522, 0.94562), new CNumber(0.19369, 0.22727), new CNumber(0.93765, 0.57145), new CNumber(0.80042, 0.34577), new CNumber(0.12658, 0.22209), new CNumber(0.79551, 0.95095), new CNumber(0.34824, 0.21561), new CNumber(0.03, 0.42283), new CNumber(0.66772, 0.17177), new CNumber(0.62885, 0.37655), new CNumber(0.86843, 0.00222), new CNumber(0.3272, 0.01863), new CNumber(0.48101, 0.13229), new CNumber(0.52672, 0.10127), new CNumber(0.09069, 0.66551), new CNumber(0.85574, 0.22091), new CNumber(0.2036, 0.26362), new CNumber(0.45319, 0.70855), new CNumber(0.8619, 0.32881), new CNumber(0.87954, 0.40211), new CNumber(0.72609, 0.54926), new CNumber(0.06173, 0.70943), new CNumber(0.249, 0.15949), new CNumber(0.34828, 0.86934), new CNumber(0.26575, 0.40123), new CNumber(0.64924, 0.59451), new CNumber(0.08251, 0.07417), new CNumber(0.44442, 0.40053), new CNumber(0.03482, 0.47076), new CNumber(0.53107, 0.28036), new CNumber(0.97127, 0.25901), new CNumber(0.8559, 0.83469), new CNumber(0.21252, 0.46033), new CNumber(0.61507, 0.92657), new CNumber(0.4268, 0.43197), new CNumber(0.44359, 0.18648), new CNumber(0.52549, 0.21405), new CNumber(0.1256, 0.7557), new CNumber(-0.0, -1.0), new CNumber(0.87474, 0.59101), new CNumber(0.95181, 0.77451)};
        aRowPointers = new int[]{0, 2, 2, 6, 7, 9, 11, 12, 13, 13, 14, 15, 17, 17, 17, 17, 19, 20, 20, 22, 23, 23, 23, 24, 25, 25, 25, 25, 26, 27, 27, 27, 28, 30, 31, 33, 36, 38, 38, 39, 40, 40, 40, 40, 41, 42, 43, 43, 44, 46, 46, 46, 46, 48, 48, 50, 51, 52, 53, 53, 54, 56, 56, 57, 57, 57, 58, 59, 60, 62, 63, 63, 63, 66, 67, 68, 69, 70, 70, 72, 75, 75, 75, 76, 78, 79, 79, 79, 80, 82, 83, 84, 85, 86, 87, 87, 88, 88, 90, 92, 92, 92, 93, 93, 93, 93, 94, 95, 95, 95, 96, 96, 96, 97, 98, 98, 99, 99, 99, 99, 100, 100, 100, 102, 102, 103, 103, 104, 104, 104, 104};
        aColIndices = new int[]{23, 115, 49, 86, 105, 125, 138, 9, 29, 1, 41, 51, 106, 138, 69, 66, 87, 125, 135, 120, 58, 123, 29, 109, 136, 148, 90, 46, 8, 61, 67, 3, 14, 59, 87, 112, 92, 97, 97, 85, 54, 45, 119, 134, 5, 126, 19, 114, 101, 128, 0, 76, 47, 110, 76, 81, 61, 142, 53, 55, 5, 63, 8, 56, 63, 66, 8, 11, 21, 44, 45, 116, 7, 17, 91, 47, 2, 125, 26, 22, 14, 109, 58, 82, 57, 117, 37, 107, 4, 27, 58, 106, 24, 93, 1, 144, 102, 144, 18, 44, 41, 125, 11, 117};
        A = new CsrCMatrixOld(aShape, aEntries, aRowPointers, aColIndices);

        expSize = 130;
        expEntries = new CNumber[]{new CNumber(15.6, -9.3), new CNumber(325.6, -0.0015), new CNumber(0.09069, 0.66551), new CNumber(-0.0, -1.0)};
        expIndices = new int[]{2, 15, 83, 122};
        exp = new CooCVectorOld(expSize, expEntries, expIndices);

        assertEquals(exp, A.getCol(125));

        // --------------------- sub-case 3 ---------------------
        A = new CsrCMatrixOld(100235, 15235);
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(-1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(15235));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(-4));
        assertThrows(IndexOutOfBoundsException.class, ()->A.getCol(222356));
    }
}
