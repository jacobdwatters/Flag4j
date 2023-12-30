package com.flag4j.complex_sparse_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class CooCMatrixMultTransposeTests {

    @Test
    void realSparseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        double[] bEntries;
        CooMatrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.18007+0.47235i"), new CNumber("0.00031+0.04818i"), new CNumber("0.16436+0.3724i")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{0, 0, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6, 5);
        bEntries = new double[]{0.23308, 0.46654, 0.34077, 0.45768, 0.76329, 0.19749, 0.91107, 0.22286, 0.8972, 0.95443};
        bRowIndices = new int[]{0, 1, 1, 1, 2, 2, 2, 3, 4, 5};
        bColIndices = new int[]{3, 0, 1, 2, 2, 3, 4, 1, 3, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0840098578+0.220370169i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0001446274+0.0224778972i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.038309028800000006+0.086798992i"), new CNumber("0.0"), new CNumber("0.0324594564+0.073545276i"), new CNumber("0.0"), new CNumber("0.147463792+0.33411728i"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.13719+0.26853i"), new CNumber("0.84603+0.45432i"), new CNumber("0.35231+0.6201i"), new CNumber("0.7042+0.79757i"), new CNumber("0.3678+0.11384i"), new CNumber("0.57107+0.78701i"), new CNumber("0.96628+0.95622i"), new CNumber("0.1145+0.88233i"), new CNumber("0.13592+0.51696i"), new CNumber("0.46483+0.8233i"), new CNumber("0.16055+0.09932i"), new CNumber("0.37551+0.79906i"), new CNumber("0.37508+0.67118i"), new CNumber("0.12633+0.87638i"), new CNumber("0.60608+0.37761i")};
        aRowIndices = new int[]{0, 1, 2, 3, 3, 3, 4, 7, 7, 7, 8, 8, 9, 9, 9};
        aColIndices = new int[]{19, 22, 8, 1, 12, 20, 1, 0, 9, 12, 17, 18, 1, 13, 14};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11, 23);
        bEntries = new double[]{0.56718, 0.61277, 0.98931, 0.27197, 0.42635, 0.604, 0.43965, 0.73371, 0.0552, 0.30879, 0.07427, 0.88552, 0.06859, 0.82031, 0.78877, 0.72343, 0.39534, 0.26539, 0.60927, 0.48236, 0.50133, 0.79897, 0.4143, 0.82129, 0.96133, 0.49847, 0.36263, 0.75414, 0.79191, 0.30996, 0.39423, 0.59391, 0.12437, 0.81227, 0.46678, 0.69816, 0.23565, 0.26345, 0.14353, 0.51491, 0.42769, 0.26879, 0.05645, 0.81837, 0.75161, 0.84688, 0.07325, 0.26145, 0.99592, 0.60897, 0.17337, 0.52817, 0.21395, 0.19048, 0.40601, 0.0891, 0.05531, 0.98314, 0.04684, 0.6489, 0.30786, 0.66835, 0.30513, 0.30018, 0.87291, 0.33619, 0.24299, 0.67784, 0.13326, 0.36638, 0.74839, 0.65227, 0.79829, 0.26297, 0.90328, 0.68587, 0.65314, 0.27191, 0.78316, 0.90408, 0.79376, 0.64697, 0.97064, 0.53179, 0.7723, 0.04114, 0.09927, 0.82622, 0.24713};
        bRowIndices = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10};
        bColIndices = new int[]{2, 4, 7, 11, 15, 18, 22, 1, 2, 4, 11, 16, 19, 21, 1, 2, 3, 4, 6, 8, 10, 19, 20, 22, 3, 10, 13, 15, 18, 20, 22, 0, 2, 3, 4, 12, 13, 17, 19, 21, 22, 3, 4, 5, 8, 9, 11, 14, 17, 20, 22, 3, 6, 9, 10, 11, 0, 3, 4, 9, 12, 15, 17, 0, 1, 2, 7, 8, 9, 10, 11, 0, 4, 5, 8, 9, 15, 17, 20, 21, 22, 0, 1, 3, 6, 7, 9, 10, 12};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber(0.0, 0.0), new CNumber(0.0094098621, 0.018418472699999997), new CNumber(0.1096106943, 0.21454741409999997), new CNumber(0.0, 0.0), new CNumber(0.0196908807, 0.038542110899999996), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.37195708949999995, 0.199741788), new CNumber(0.0, 0.0), new CNumber(0.6948359786999999, 0.3731284728), new CNumber(0.3335304069, 0.1791065736), new CNumber(0.3618385707, 0.1943081208), new CNumber(0.14667622109999998, 0.0787654584), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.6715447728, 0.3606210432), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.1699402516, 0.299111436), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.2647997191, 0.46607336099999996), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.23880981040000002, 0.420328584), new CNumber(0.3182345768, 0.560123928), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.516678582, 0.5851850847), new CNumber(0.792046135, 0.9551575319), new CNumber(0.1770088572, 0.24394161960000002), new CNumber(0.256783248, 0.0794785344), new CNumber(0.34776449789999997, 0.4792654797), new CNumber(0.0, 0.0), new CNumber(0.11323090800000002, 0.035046782400000004), new CNumber(0.614703222, 0.6962068287), new CNumber(0.44723918119999995, 0.6163547516), new CNumber(0.774419102, 0.802286624)},
                {new CNumber(0.0, 0.0), new CNumber(0.7089692988, 0.7015881761999999), new CNumber(0.7621726756, 0.7542376493999999), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.8434754748, 0.8346940002), new CNumber(0.0, 0.0), new CNumber(0.9379100192, 0.9281453807999999)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.39252840780000003, 1.0988197383), new CNumber(0.1151079296, 0.43780308479999996), new CNumber(0.025890041600000004, 0.0984705408), new CNumber(0.23763404680000005, 0.6377181543), new CNumber(0.0524833092, 0.33374790899999995), new CNumber(0.1679083654, 0.9300847443), new CNumber(0.20244428130000003, 0.8256217883000001)},
                {new CNumber(0.22680804, 0.48263224), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.2973701241, 0.6327836046), new CNumber(0.0422968975, 0.026165854000000002), new CNumber(0.159894956, 0.0989147744), new CNumber(0.0, 0.0), new CNumber(0.0489886215, 0.0303055116), new CNumber(0.0, 0.0), new CNumber(0.043655150499999996, 0.0270061012), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.2751999468, 0.4924514778), new CNumber(0.2958518516, 0.5294066486), new CNumber(0.0458110479, 0.31780167940000004), new CNumber(0.0297696645, 0.206518947), new CNumber(0.158459616, 0.0987261345), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.32741108280000003, 0.5858797338), new CNumber(0.0, 0.0), new CNumber(0.3640676512, 0.6514741552)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)}
        };

        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.50232+0.68872i"), new CNumber("0.78106+0.68678i"), new CNumber("0.48471+0.12952i"), new CNumber("0.22888+0.15437i")};
        aRowIndices = new int[]{0, 1, 4, 4};
        aColIndices = new int[]{2, 1, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new double[]{0.05099};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0247153629+0.0066042248i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.10097+0.56982i"), new CNumber("0.58427+0.12349i"), new CNumber("0.56427+0.85603i"), new CNumber("0.43605+0.51715i")};
        aRowIndices = new int[]{0, 2, 3, 4};
        aColIndices = new int[]{0, 1, 0, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new double[]{0.41193, 0.15502};
        bRowIndices = new int[]{0, 1};
        bColIndices = new int[]{0, 1};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0415925721+0.2347259526i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.09057353539999999+0.0191434198i")},
                {new CNumber("0.23243974110000004+0.3526244379i"), new CNumber("0.0")},
                {new CNumber("0.1796220765+0.2130295995i"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.69801+0.91492i"), new CNumber("0.32724+0.32501i"), new CNumber("0.28081+0.8828i"), new CNumber("0.64801+0.49649i")};
        aRowIndices = new int[]{1, 1, 3, 4};
        aColIndices = new int[]{0, 1, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new double[]{0.84814, 0.0606, 0.27968, 0.31826};
        bRowIndices = new int[]{0, 1, 2, 4};
        bColIndices = new int[]{0, 1, 0, 0};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix final0a = a;
        CooMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.68571+0.35483i"), new CNumber("0.92183+0.80883i"), new CNumber("0.58557+0.25692i"), new CNumber("0.79019+0.75866i")};
        aRowIndices = new int[]{0, 1, 3, 4};
        aColIndices = new int[]{0, 0, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.38021, 0.2587, 0.1574, 0.11695, 0.46968, 0.20872, 0.1969, 0.2504, 0.33862};
        bRowIndices = new int[]{0, 0, 0, 1, 1, 3, 3, 4, 4};
        bColIndices = new int[]{1, 2, 3, 0, 1, 2, 4, 2, 3};
        b = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix final1a = a;
        CooMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }


    @Test
    void complexSparseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        CNumber[] bEntries;
        CooCMatrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.13333+0.54407i"), new CNumber("0.28186+0.80017i"), new CNumber("0.27979+0.90149i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6, 5);
        bEntries = new CNumber[]{new CNumber("0.7783+0.53458i"), new CNumber("0.90399+0.64094i"), new CNumber("0.51196+0.51884i"), new CNumber("0.46424+0.32518i"), new CNumber("0.69994+0.96631i"), new CNumber("0.89585+0.321i"), new CNumber("0.29658+0.1626i"), new CNumber("0.70076+0.3312i"), new CNumber("0.35692+0.11799i"), new CNumber("0.83837+0.80204i")};
        bRowIndices = new int[]{0, 1, 2, 3, 4, 4, 4, 4, 5, 5};
        bColIndices = new int[]{4, 0, 2, 1, 0, 1, 3, 4, 0, 3};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("-0.20838324060000005+0.7734490298000001i"), new CNumber("0.0"), new CNumber("-0.21402565199999998+0.3477190144i"), new CNumber("0.0"), new CNumber("-0.06750009040000002+0.6540791612000001i"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("-0.32487363849999995+0.9942665476999999i"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-0.6752825893000001+0.9013527855000001i"), new CNumber("-0.006504158300000007+0.35477223290000004i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.1091+0.67734i"), new CNumber("0.86221+0.18731i"), new CNumber("0.86577+0.3935i"), new CNumber("0.79207+0.16091i"), new CNumber("0.15273+0.23584i"), new CNumber("0.40546+0.55037i"), new CNumber("0.06335+0.55614i"), new CNumber("0.45687+0.32064i"), new CNumber("0.84184+0.56079i"), new CNumber("0.53839+0.56551i"), new CNumber("0.82059+0.48871i"), new CNumber("0.19813+0.82157i"), new CNumber("0.23558+0.75971i"), new CNumber("0.89675+0.70556i"), new CNumber("0.30268+0.21512i")};
        aRowIndices = new int[]{0, 0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 7, 8, 10, 10};
        aColIndices = new int[]{2, 7, 8, 0, 10, 3, 14, 17, 2, 22, 13, 14, 0, 4, 20};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11, 23);
        bEntries = new CNumber[]{new CNumber("0.09315+0.56922i"), new CNumber("0.11107+0.98862i"), new CNumber("0.32474+0.35843i"), new CNumber("0.79492+0.42901i"), new CNumber("0.82111+0.21961i"), new CNumber("0.58618+0.95779i"), new CNumber("0.05473+0.03025i"), new CNumber("0.97621+0.86671i"), new CNumber("0.80408+0.59324i"), new CNumber("0.03767+0.67623i"), new CNumber("0.61841+0.27712i"), new CNumber("0.88397+0.64611i"), new CNumber("0.56549+0.54455i"), new CNumber("0.09233+0.22337i"), new CNumber("0.41633+0.25273i"), new CNumber("0.56249+0.55354i"), new CNumber("0.62517+0.56234i"), new CNumber("0.69762+0.90077i"), new CNumber("0.89853+0.70042i"), new CNumber("0.79632+0.18535i"), new CNumber("0.22337+0.65651i"), new CNumber("0.98515+0.5477i"), new CNumber("0.23836+0.22636i"), new CNumber("0.37038+0.87641i"), new CNumber("0.49897+0.72714i"), new CNumber("0.38397+0.13316i"), new CNumber("0.05107+0.31637i"), new CNumber("0.26104+0.52205i"), new CNumber("0.30751+0.82356i"), new CNumber("0.43528+0.9727i"), new CNumber("0.29987+0.92454i"), new CNumber("0.42656+0.15536i"), new CNumber("0.76042+0.01935i"), new CNumber("0.46552+0.85124i"), new CNumber("0.52773+0.92759i"), new CNumber("0.11239+0.98725i"), new CNumber("0.3875+0.27288i"), new CNumber("0.52375+0.34944i"), new CNumber("0.49127+0.69019i"), new CNumber("0.45659+0.25621i"), new CNumber("0.18231+0.12218i"), new CNumber("0.22033+0.51241i"), new CNumber("0.83535+0.08698i"), new CNumber("0.12028+0.55878i"), new CNumber("0.06383+0.67314i"), new CNumber("0.70707+0.31567i"), new CNumber("0.70939+0.15072i"), new CNumber("0.16456+0.52904i"), new CNumber("0.98872+0.05189i"), new CNumber("0.71359+0.01168i"), new CNumber("0.88549+0.86375i"), new CNumber("0.09344+0.08053i"), new CNumber("0.72458+0.6146i"), new CNumber("0.69591+0.15405i"), new CNumber("0.08155+0.51061i"), new CNumber("0.09773+0.01328i"), new CNumber("0.51441+0.35341i"), new CNumber("0.88777+0.58754i"), new CNumber("0.35335+0.27309i"), new CNumber("0.40736+0.04797i"), new CNumber("0.21963+0.78092i"), new CNumber("0.5233+0.52295i"), new CNumber("0.4413+0.23607i"), new CNumber("0.41783+0.78562i"), new CNumber("0.69526+0.53609i"), new CNumber("0.81308+0.77482i"), new CNumber("0.08673+0.26201i"), new CNumber("0.6984+0.37514i"), new CNumber("0.17061+0.67571i"), new CNumber("0.28942+0.7071i"), new CNumber("0.83855+0.65238i"), new CNumber("0.49973+0.6632i"), new CNumber("0.45344+0.4296i"), new CNumber("0.54924+0.01977i"), new CNumber("0.86923+0.24688i"), new CNumber("0.82967+0.21481i"), new CNumber("0.16618+0.51186i"), new CNumber("0.03249+0.94928i"), new CNumber("0.11377+0.88986i"), new CNumber("0.47167+0.20325i"), new CNumber("0.34062+0.91378i"), new CNumber("0.60451+0.33083i"), new CNumber("0.09217+0.30043i"), new CNumber("0.24986+0.79771i"), new CNumber("0.73108+0.74435i"), new CNumber("0.01894+0.89016i"), new CNumber("0.21863+0.62992i"), new CNumber("0.8874+0.15921i"), new CNumber("0.34217+0.47171i")};
        bRowIndices = new int[]{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10};
        bColIndices = new int[]{4, 5, 11, 21, 22, 3, 4, 8, 9, 11, 12, 14, 19, 20, 3, 5, 6, 8, 9, 12, 14, 16, 17, 19, 22, 3, 7, 18, 22, 0, 4, 5, 6, 10, 14, 15, 17, 21, 22, 4, 5, 10, 11, 12, 16, 17, 19, 20, 2, 6, 7, 9, 12, 14, 15, 17, 19, 2, 5, 7, 8, 9, 10, 12, 16, 19, 20, 22, 2, 10, 15, 18, 20, 2, 3, 5, 6, 14, 16, 17, 18, 21, 6, 8, 15, 16, 19, 21, 22};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber(0.0, 0.0), new CNumber(0.5041229467000001, 1.1345101517), new CNumber(0.24952547240000006, 1.0543731129), new CNumber(-0.015226200000000002, 0.28234329939999997), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.6744114998, 1.5859558232000002), new CNumber(-0.07600698659999994, 1.5456070745000001), new CNumber(-0.43907186040000007, 0.18928093840000002), new CNumber(0.046531072199999995, 0.37417912859999997), new CNumber(-0.0975775928, 0.7889532967000001)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.058597500600000046, 1.0802855158), new CNumber(-0.08719577350000002, 0.13022300650000002), new CNumber(0.0, 0.0), new CNumber(0.011725000200000009, 0.1401311631), new CNumber(-0.12255934739999996, 0.1762521958), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(-0.5927944554, 1.2435035643), new CNumber(-0.2849313674, 0.6772671918), new CNumber(0.08239720699999997, 0.2653166225), new CNumber(-0.39289732529999993, 0.6011732743), new CNumber(0.22182264210000002, 0.3709350777), new CNumber(-0.0011956625999999804, 0.43418583569999997), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(-0.15898989460000004, 0.9007990528), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.8032446517, 0.5981473664), new CNumber(0.41787374019999995, 0.9924672119), new CNumber(-0.23530508850000004, 0.6645160883000001), new CNumber(0.45128538329999995, 0.32465147639999997), new CNumber(0.0, 0.0)},
                {new CNumber(0.31788576180000006, 0.582581744), new CNumber(0.0, 0.0), new CNumber(-0.14256448309999992, 0.6736574292999999), new CNumber(-0.3001711066999999, 0.6172964485), new CNumber(-0.12581449159999997, 0.6494094918), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.16386615460000004, 0.5969238085999999), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(-0.08253581580000002, 0.4474645036)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(-0.3556836165999999, 0.8542570072), new CNumber(-0.4951126226000001, 0.3135884172), new CNumber(0.0, 0.0), new CNumber(-0.6575209714000001, 0.6173505428), new CNumber(0.0, 0.0), new CNumber(0.011317789800000033, 0.6022607052), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(-0.7734627259000001, 0.2147736557), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(-0.6364266546, 0.5598352348), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(-0.31808660069999994, 0.576170949), new CNumber(0.007631027499999998, 0.1532136475), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(-0.3834100199, 1.0406575222), new CNumber(0.16467749090000003, 0.7474379323), new CNumber(0.0, 0.0), new CNumber(-0.030112154800000006, 0.09796254440000002), new CNumber(0.044831667199999994, 0.22757534080000003), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)}
        };
        exp = new CMatrix(expEntries);

        CMatrix act = a.multTranspose(b);

        assertTrue(exp.allClose(act));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.84627+0.52397i"), new CNumber("0.65009+0.84477i"), new CNumber("0.66607+0.81437i"), new CNumber("0.1052+0.24359i")};
        aRowIndices = new int[]{0, 2, 3, 3};
        aColIndices = new int[]{0, 1, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new CNumber[]{new CNumber("0.78083+0.73775i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.02414+0.82288i"), new CNumber("0.94529+0.19391i"), new CNumber("0.1772+0.50064i"), new CNumber("0.09017+0.45946i")};
        aRowIndices = new int[]{2, 3, 3, 3};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new CNumber[]{new CNumber("0.65184+0.97107i"), new CNumber("0.70615+0.94007i")};
        bRowIndices = new int[]{1, 1};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("-0.783338664+0.5598277289999999i")},
                {new CNumber("0.0"), new CNumber("0.08237078510000007+1.5644483947i")},
                {new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.43846+0.30964i"), new CNumber("0.76753+0.33258i"), new CNumber("0.92751+0.69248i"), new CNumber("0.18901+0.72498i")};
        aRowIndices = new int[]{0, 1, 1, 4};
        aColIndices = new int[]{1, 1, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new CNumber[]{new CNumber("0.18351+0.73322i"), new CNumber("0.59183+0.61502i"), new CNumber("0.1071+0.90921i"), new CNumber("0.34189+0.85244i")};
        bRowIndices = new int[]{0, 2, 3, 4};
        bColIndices = new int[]{1, 1, 1, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix final0a = a;
        CooCMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.65534+0.28091i"), new CNumber("0.29233+0.20161i"), new CNumber("0.10737+0.28494i"), new CNumber("0.24774+0.02259i")};
        aRowIndices = new int[]{1, 3, 3, 4};
        aColIndices = new int[]{0, 0, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new CNumber[]{new CNumber("0.27254+0.44057i"), new CNumber("0.49021+0.16341i"), new CNumber("0.14641+0.73422i"), new CNumber("0.90959+0.99102i"), new CNumber("0.46919+0.61656i"), new CNumber("0.79621+0.30459i"), new CNumber("0.36651+0.7489i"), new CNumber("0.01621+0.51064i"), new CNumber("0.3416+0.99506i")};
        bRowIndices = new int[]{0, 0, 1, 2, 2, 2, 2, 3, 3};
        bColIndices = new int[]{3, 4, 0, 1, 2, 3, 4, 2, 4};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix final1a = a;
        CooCMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }


    @Test
    void realDenseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        double[][] bEntries;
        Matrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.69195+0.39492i"), new CNumber("0.5457+0.81949i"), new CNumber("0.03667+0.05843i")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{0, 1, 3};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.75059, 0.33636, 0.1625, 0.15499, 0.65849},
                {0.18386, 0.89345, 0.68082, 0.47839, 0.83208},
                {0.99886, 0.47874, 0.73715, 0.56744, 0.82517},
                {0.18916, 0.44241, 0.85278, 0.95108, 0.06971},
                {0.88923, 0.46296, 0.1603, 0.66721, 0.02301},
                {0.95954, 0.0763, 0.88728, 0.31218, 0.14393}};
        b = new Matrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.5193707504999999+0.2964230028i"), new CNumber("0.12722192699999998+0.0726099912i"), new CNumber("0.6911611769999999+0.3944697912i"), new CNumber("0.13088926199999998+0.0747030672i"), new CNumber("0.6153026985+0.35117471159999997i"), new CNumber("0.6639537029999999+0.3789415368i")},
                {new CNumber("0.18355165199999998+0.2756436564i"), new CNumber("0.48755566499999997+0.7321733405i"), new CNumber("0.261248418+0.39232264260000005i"), new CNumber("0.241423137+0.36255057090000004i"), new CNumber("0.25263727199999997+0.3793910904i"), new CNumber("0.04163691+0.06252708700000001i")},
                {new CNumber("0.0056834833+0.0090560657i"), new CNumber("0.0175425613+0.0279523277i"), new CNumber("0.0208080248+0.033155519200000004i"), new CNumber("0.0348761036+0.0555716044i"), new CNumber("0.0244665907+0.0389850803i"), new CNumber("0.011447640600000001+0.018240677400000002i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.38493+0.69137i"), new CNumber("0.0823+0.96975i"), new CNumber("0.52345+0.70595i"), new CNumber("0.74239+0.36901i"), new CNumber("0.1443+0.56394i"), new CNumber("0.91757+0.50549i"), new CNumber("0.27492+0.17729i"), new CNumber("0.04249+0.03255i"), new CNumber("0.05289+0.44651i"), new CNumber("0.0021+0.26898i"), new CNumber("0.86822+0.58018i"), new CNumber("0.60689+0.55886i"), new CNumber("0.26975+0.33019i"), new CNumber("0.94561+0.18236i"), new CNumber("0.49722+0.7081i")};
        aRowIndices = new int[]{0, 0, 1, 1, 2, 2, 3, 4, 4, 6, 6, 7, 7, 8, 8};
        aColIndices = new int[]{0, 2, 7, 21, 7, 15, 14, 0, 7, 11, 22, 6, 12, 4, 12};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.39875, 0.2222, 0.87044, 0.79841, 0.05193, 0.87607, 0.97317, 0.3794, 0.64869, 0.47226, 0.84398, 0.31248, 0.77409, 0.94563, 0.55833, 0.86353, 0.47604, 0.39289, 0.02278, 0.53816, 0.37658, 0.46545, 0.20555},
                {0.18562, 0.66036, 0.90864, 0.16813, 0.50127, 0.12898, 0.49088, 0.07396, 0.35174, 0.15226, 0.02929, 0.41453, 0.87492, 0.97186, 0.41701, 0.31137, 0.32914, 0.54553, 0.59213, 0.41067, 0.86016, 0.96245, 0.5232},
                {0.66969, 0.83251, 0.47705, 0.69806, 0.48698, 0.25965, 0.96812, 0.3254, 0.31153, 0.48704, 0.95447, 0.47382, 0.15288, 0.24298, 0.38779, 0.56684, 0.39734, 0.196, 0.43591, 0.51018, 0.91778, 0.76632, 0.70973},
                {0.96171, 0.06687, 0.86592, 0.14168, 0.68569, 0.0192, 0.22838, 0.75406, 0.10151, 0.05939, 0.50425, 0.6834, 0.08845, 0.70586, 0.52355, 0.51948, 0.03677, 0.86775, 0.92875, 0.61959, 0.28069, 0.71145, 0.41433},
                {0.85827, 0.25423, 0.54481, 0.75977, 0.92943, 0.97883, 0.90156, 0.0952, 0.50752, 0.98634, 0.68365, 0.67421, 0.45627, 0.90465, 0.98381, 0.65696, 0.20766, 0.42459, 0.84673, 0.21907, 0.91359, 0.01018, 0.27154},
                {0.36939, 0.88146, 0.88092, 0.3715, 0.74723, 0.10591, 0.76083, 0.25742, 0.49128, 0.95475, 0.96115, 0.85741, 0.54404, 0.49643, 0.19024, 0.87034, 0.54688, 0.02171, 0.81189, 0.9437, 0.23632, 0.08158, 0.51796},
                {0.12907, 0.08227, 0.19712, 0.37813, 0.53306, 0.12747, 0.39995, 0.85421, 0.76769, 0.21097, 0.93637, 0.15814, 0.50745, 0.35202, 0.98314, 0.76853, 0.21373, 0.97258, 0.2507, 0.04802, 0.34327, 0.56543, 0.69559},
                {0.74226, 0.30454, 0.64889, 0.71143, 0.31354, 0.79529, 0.71604, 0.02694, 0.76273, 0.6413, 0.83401, 0.67861, 0.04476, 0.55584, 0.68212, 0.44486, 0.1122, 0.59327, 0.6448, 0.56276, 0.64185, 0.81826, 0.09657},
                {0.84769, 0.10098, 0.21177, 0.11168, 0.659, 0.5763, 0.60147, 0.78667, 0.04956, 0.50817, 0.68706, 0.02883, 0.92889, 0.7675, 0.83614, 0.8543, 0.97858, 0.60112, 0.87719, 0.24513, 0.28658, 0.70932, 0.02847},
                {0.00234, 0.70009, 0.78097, 0.65756, 0.84871, 0.48446, 0.12295, 0.15414, 0.04519, 0.206, 0.05482, 0.95631, 0.50146, 0.09516, 0.93109, 0.74706, 0.78342, 0.50685, 0.18246, 0.71533, 0.60555, 0.62308, 0.63251},
                {0.11011, 0.38167, 0.25484, 0.17403, 0.13975, 0.92719, 0.50381, 0.30755, 0.28791, 0.32077, 0.75766, 0.234, 0.25922, 0.46057, 0.48356, 0.54309, 0.14395, 0.61734, 0.83765, 0.57755, 0.07818, 0.25984, 0.04551}};
        b = new Matrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.22512804949999998+1.1197929775i"), new CNumber("0.1462317786+1.0094857394i"), new CNumber("0.2970449867+0.9256228128i"), new CNumber("0.4414562463+1.5046233626999999i"), new CNumber("0.37521173409999997+1.1217116274i"), new CNumber("0.21468900870000002+1.1096573343i"), new CNumber("0.0659058911+0.2803922459i"), new CNumber("0.3391217888+1.1424373737i"), new CNumber("0.34372998270000005+0.7914313928000001i"), new CNumber("0.0651745672+0.7589634633000001i"), new CNumber("0.0633579743+0.3232578407i")},
                {new CNumber("0.5441423555+0.4395931345i"), new CNumber("0.7532276175+0.4073657365i"), new CNumber("0.7392389348+0.5124958732i"), new CNumber("0.9228860725+0.7948608214999999i"), new CNumber("0.0573899702+0.0709629618i"), new CNumber("0.1953106752+0.2118294848i"), new CNumber("0.8669058022+0.8116788738i"), new CNumber("0.6215697844+0.3209644156i"), new CNumber("0.9383744862999999+0.8170958597i"), new CNumber("0.5432529441999999+0.3387378838i"), new CNumber("0.3538896651+0.3129984809i")},
                {new CNumber("0.8470966421+0.6504646157i"), new CNumber("0.2963761989+0.19910342369999998i"), new CNumber("0.5670705988+0.4700380276i"), new CNumber("0.5854701216+0.6878365416i"), new CNumber("0.6165441472000001+0.3857737984i"), new CNumber("0.8357435798+0.5851176014i"), new CNumber("0.8284425751+0.8702074171i"), new CNumber("0.4120776322+0.24006482499999998i"), new CNumber("0.8973965319999999+0.8754747867999999i"), new CNumber("0.7077222462+0.464557071i"), new CNumber("0.5427025563+0.44796631109999996i")},
                {new CNumber("0.1534960836+0.0989863257i"), new CNumber("0.1146443892+0.0739317029i"), new CNumber("0.1066112268+0.0687512891i"), new CNumber("0.14393436599999998+0.09282017949999999i"), new CNumber("0.2704690452+0.1744196749i"), new CNumber("0.052300780799999995+0.0337276496i"), new CNumber("0.2702848488+0.1743008906i"), new CNumber("0.18752843039999997+0.1209330548i"), new CNumber("0.2298716088+0.1482392606i"), new CNumber("0.2559752628+0.1650729461i"), new CNumber("0.1329403152+0.0857303524i")},
                {new CNumber("0.0370093535+0.1823852065i"), new CNumber("0.0117987382+0.0390658106i"), new CNumber("0.0456655341+0.16709276350000002i"), new CNumber("0.0807452913+0.3679989911i"), new CNumber("0.0415030203+0.07044444050000001i"), new CNumber("0.029310324899999997+0.1269642487i"), new CNumber("0.0506633512+0.38561453560000003i"), new CNumber("0.032963484+0.036189542400000003i"), new CNumber("0.0776253244+0.37884833120000005i"), new CNumber("0.0082518912+0.0689012184i"), new CNumber("0.0209448934+0.140908231i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.179118829+0.2033068694i"), new CNumber("0.455123217+0.4150504554i"), new CNumber("0.6171968026+0.539219255i"), new CNumber("0.3611647326+0.4242069114i"), new CNumber("0.2371722998+0.33889108300000004i"), new CNumber("0.4515037922+0.5311361746i"), new CNumber("0.6042572438+0.44610390340000006i"), new CNumber("0.0852690864+0.23856050040000001i"), new CNumber("0.0247787664+0.024272417999999997i"), new CNumber("0.5511660832+0.6241979155999999i"), new CNumber("0.0400040922+0.08934531180000001i")},
                {new CNumber("0.7994179188+0.7994625633i"), new CNumber("0.5339198332+0.5632230316i"), new CNumber("0.6287817268+0.5915229904i"), new CNumber("0.1624609257+0.1568377523i"), new CNumber("0.6702265809+0.6545016129000001i"), new CNumber("0.6084949087+0.6048340214i"), new CNumber("0.37961029300000004+0.3910709725i"), new CNumber("0.4466315256000001+0.4149454188i"), new CNumber("0.6155942057999999+0.6428477132999999i"), new CNumber("0.20988596050000002+0.2342889144i"), new CNumber("0.3756818459+0.3671511084i")},
                {new CNumber("0.43399855709999996+0.5576030837999999i"), new CNumber("0.9090336471+0.7109424492i"), new CNumber("0.5365081514+0.19706000079999997i"), new CNumber("0.6923744299+0.18767387339999997i"), new CNumber("1.1057448717+0.49257564179999996i"), new CNumber("0.9770957291+0.5214995868i"), new CNumber("0.7563811556+0.45653416659999996i"), new CNumber("0.31874212659999995+0.0888717104i"), new CNumber("1.0850196758+0.777922249i"), new CNumber("1.0518846042999999+0.5098545816i"), new CNumber("0.2610383659+0.209038492i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.78326+0.38303i"), new CNumber("0.91661+0.74583i"), new CNumber("0.66673+0.6148i"), new CNumber("0.12376+0.32819i")};
        aRowIndices = new int[]{0, 1, 1, 3};
        aColIndices = new int[]{2, 0, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.22966, 0.86485, 0.33411}};
        b = new Matrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.2616949986+0.1279741533i")},
                {new CNumber("0.7871300931+0.7029970978i")},
                {new CNumber("0.0")},
                {new CNumber("0.107033836+0.2838351215i")},
                {new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.11009+0.86316i"), new CNumber("0.51338+0.73432i"), new CNumber("0.36883+0.97426i"), new CNumber("0.15692+0.24652i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.76359, 0.30769, 0.53053},
                {0.65722, 0.39491, 0.2375}};
        b = new Matrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0840636231+0.6591003444i"), new CNumber("0.0723533498+0.5672860152i")},
                {new CNumber("0.5054971369+0.8604894681999999i"), new CNumber("0.4830582589+0.867354807i")},
                {new CNumber("0.0832507676+0.13078625559999998i"), new CNumber("0.037268499999999996+0.058548499999999996i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.77528+0.60978i"), new CNumber("0.10935+0.98871i"), new CNumber("0.62819+0.1926i"), new CNumber("0.46678+0.6006i")};
        aRowIndices = new int[]{2, 2, 3, 4};
        aColIndices = new int[]{1, 2, 2, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.14513, 0.63269},
                {0.65445, 0.20141},
                {0.3549, 0.72303},
                {0.34459, 0.3964},
                {0.67877, 0.48217}};
        b = new Matrix(bEntries);

        CooCMatrix final0a = a;
        Matrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.11764+0.52728i"), new CNumber("0.23086+0.21765i"), new CNumber("0.55925+0.12422i"), new CNumber("0.75233+0.19753i")};
        aRowIndices = new int[]{0, 0, 1, 2};
        aColIndices = new int[]{0, 2, 1, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.99268, 0.70715, 0.89663, 0.20563, 0.02976},
                {0.68286, 0.71161, 0.04641, 0.90821, 0.40791},
                {0.09486, 0.76135, 0.27702, 0.81195, 0.77483},
                {0.32367, 0.23858, 0.09242, 0.11179, 0.98554},
                {0.71075, 0.12146, 0.43615, 0.12831, 0.47346}};
        b = new Matrix(bEntries);

        CooCMatrix final1a = a;
        Matrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }


    @Test
    void complexDenseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        CooCMatrix a;

        CNumber[][] bEntries;
        CMatrix b;

        CNumber[][] expEntries;
        CMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.87979+0.41297i"), new CNumber("0.36913+0.04991i"), new CNumber("0.84641+0.37285i")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.24929+0.36257i"), new CNumber("0.25118+0.1858i"), new CNumber("0.51754+0.5778i"), new CNumber("0.33593+0.47237i"), new CNumber("0.48676+0.53026i")},
                {new CNumber("0.33536+0.61251i"), new CNumber("0.53747+0.13535i"), new CNumber("0.68335+0.61201i"), new CNumber("0.97624+0.24274i"), new CNumber("0.26388+0.69334i")},
                {new CNumber("0.88823+0.3955i"), new CNumber("0.99171+0.80495i"), new CNumber("0.40287+0.6422i"), new CNumber("0.15661+0.78879i"), new CNumber("0.1881+0.51764i")},
                {new CNumber("0.57397+0.66484i"), new CNumber("0.59526+0.71449i"), new CNumber("0.30769+0.32382i"), new CNumber("0.33206+0.89095i"), new CNumber("0.66623+0.02833i")},
                {new CNumber("0.58741+0.85023i"), new CNumber("0.43104+0.62956i"), new CNumber("0.85297+0.10275i"), new CNumber("0.84224+0.32519i"), new CNumber("0.37797+0.93445i")},
                {new CNumber("0.97916+0.53569i"), new CNumber("0.0915+0.39113i"), new CNumber("0.94915+0.73352i"), new CNumber("0.08018+0.89577i"), new CNumber("0.39803+0.47572i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.21671245059999997+0.7220711558i"), new CNumber("0.34846272679999996+0.8206433274i"), new CNumber("0.08923166329999999+0.7313743618999999i"), new CNumber("0.1369746397+0.4119603371i"), new CNumber("0.7080018088+0.4426494434i"), new CNumber("0.5321309241000001+1.0373140363i")},
                {new CNumber("0.08344479540000001+0.0811207478i"), new CNumber("0.1916409826+0.07678687320000001i"), new CNumber("0.3258948578+0.3466274396000001i"), new CNumber("0.18406812790000002+0.2934491203i"), new CNumber("0.12768845559999997+0.25390268920000003i"), new CNumber("0.0142540967+0.14894458189999998i")},
                {new CNumber("0.0758173244+0.3998306502i"), new CNumber("0.05547770409999997+0.6434735651i"), new CNumber("0.6043445793+0.6659317105i"), new CNumber("0.23792835369999996+0.7767319389i"), new CNumber("0.18018144259999996+0.9386589928i"), new CNumber("0.6290387990999999+0.8184931789000001i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new CNumber[]{new CNumber("0.41912+0.76956i"), new CNumber("0.51999+0.28967i"), new CNumber("0.8994+0.36263i"), new CNumber("0.62029+0.46259i"), new CNumber("0.96098+0.81581i"), new CNumber("0.18664+0.74075i"), new CNumber("0.07323+0.92791i"), new CNumber("0.41331+0.20244i"), new CNumber("0.01758+0.41576i"), new CNumber("0.4204+0.69024i"), new CNumber("0.12449+0.51214i"), new CNumber("0.72566+0.75397i"), new CNumber("0.96453+0.83956i"), new CNumber("0.12497+0.42527i"), new CNumber("0.51741+0.38359i")};
        aRowIndices = new int[]{0, 1, 2, 3, 4, 4, 6, 6, 7, 7, 8, 8, 8, 9, 9};
        aColIndices = new int[]{16, 19, 1, 7, 2, 16, 4, 22, 4, 17, 4, 15, 20, 13, 17};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.8322+0.35878i"), new CNumber("0.47371+0.35255i"), new CNumber("0.16613+0.13004i"), new CNumber("0.58945+0.91065i"), new CNumber("0.10399+0.16885i"), new CNumber("0.35431+0.01901i"), new CNumber("0.61456+0.1471i"), new CNumber("0.48457+0.43136i"), new CNumber("0.42392+0.65523i"), new CNumber("0.87141+0.31108i"), new CNumber("0.12679+0.57543i"), new CNumber("0.33716+0.92421i"), new CNumber("0.32441+0.02666i"), new CNumber("0.16055+0.24761i"), new CNumber("0.79206+0.09369i"), new CNumber("0.26826+0.48842i"), new CNumber("0.16527+0.48773i"), new CNumber("0.69281+0.55448i"), new CNumber("0.81442+0.54125i"), new CNumber("0.42845+0.17077i"), new CNumber("0.24495+0.16031i"), new CNumber("0.36959+0.42914i"), new CNumber("0.61635+0.37016i")},
                {new CNumber("0.42256+0.5206i"), new CNumber("0.13572+0.39044i"), new CNumber("0.07962+0.39612i"), new CNumber("0.61008+0.00588i"), new CNumber("0.58451+0.42492i"), new CNumber("0.53231+0.60107i"), new CNumber("0.93195+0.53278i"), new CNumber("0.61693+0.63781i"), new CNumber("0.77048+0.52765i"), new CNumber("0.35778+0.89705i"), new CNumber("0.92595+0.13264i"), new CNumber("0.66983+0.57941i"), new CNumber("0.90432+0.48925i"), new CNumber("0.49826+0.29397i"), new CNumber("0.40861+0.22671i"), new CNumber("0.14617+0.22897i"), new CNumber("0.89218+0.03271i"), new CNumber("0.44979+0.23329i"), new CNumber("0.84501+0.10084i"), new CNumber("0.65179+0.69096i"), new CNumber("0.02912+0.31611i"), new CNumber("0.14998+0.94966i"), new CNumber("0.16051+0.80502i")},
                {new CNumber("0.03331+0.33627i"), new CNumber("0.36273+0.93595i"), new CNumber("0.74426+0.43286i"), new CNumber("0.15518+0.15691i"), new CNumber("0.62696+0.07461i"), new CNumber("0.76378+0.15831i"), new CNumber("0.8804+0.39513i"), new CNumber("0.60802+0.44005i"), new CNumber("0.04202+0.08285i"), new CNumber("0.71461+0.58192i"), new CNumber("0.79868+0.87502i"), new CNumber("0.52157+0.67928i"), new CNumber("0.31179+0.22349i"), new CNumber("0.23491+0.57964i"), new CNumber("0.7386+0.92216i"), new CNumber("0.83856+0.81752i"), new CNumber("0.99382+0.34532i"), new CNumber("0.96309+0.72639i"), new CNumber("0.3965+0.81326i"), new CNumber("0.34958+0.63246i"), new CNumber("0.80841+0.70145i"), new CNumber("0.6984+0.7817i"), new CNumber("0.27723+0.21709i")},
                {new CNumber("0.74354+0.36274i"), new CNumber("0.27348+0.91678i"), new CNumber("0.62609+0.09958i"), new CNumber("0.21003+0.44538i"), new CNumber("0.55106+0.69592i"), new CNumber("0.73285+0.00077i"), new CNumber("0.99154+0.49163i"), new CNumber("0.80116+0.24268i"), new CNumber("0.58592+0.44986i"), new CNumber("0.98549+0.41368i"), new CNumber("0.62278+0.01156i"), new CNumber("0.44868+0.609i"), new CNumber("0.30024+0.79167i"), new CNumber("0.02925+0.66186i"), new CNumber("0.67699+0.73227i"), new CNumber("0.94323+0.73471i"), new CNumber("0.28488+0.59732i"), new CNumber("0.0998+0.01316i"), new CNumber("0.86811+0.44714i"), new CNumber("0.40312+0.27509i"), new CNumber("0.24252+0.25792i"), new CNumber("0.38661+0.85297i"), new CNumber("0.80331+0.61156i")},
                {new CNumber("0.62827+0.72684i"), new CNumber("0.81467+0.89771i"), new CNumber("0.42593+0.45929i"), new CNumber("0.80424+0.525i"), new CNumber("0.95065+0.61942i"), new CNumber("0.30197+0.3462i"), new CNumber("0.57861+0.77081i"), new CNumber("0.65076+0.98374i"), new CNumber("0.10544+0.75221i"), new CNumber("0.02919+0.03849i"), new CNumber("0.48367+0.03475i"), new CNumber("0.32865+0.63767i"), new CNumber("0.06525+0.54283i"), new CNumber("0.64794+0.30905i"), new CNumber("0.47948+0.74059i"), new CNumber("0.86158+0.56123i"), new CNumber("0.52088+0.85921i"), new CNumber("0.81167+0.074i"), new CNumber("0.2695+0.90389i"), new CNumber("0.21566+0.2883i"), new CNumber("0.92287+0.13942i"), new CNumber("0.71908+0.03391i"), new CNumber("0.07065+0.20166i")},
                {new CNumber("0.00755+0.94602i"), new CNumber("0.52168+0.70259i"), new CNumber("0.0367+0.20927i"), new CNumber("0.31163+0.59011i"), new CNumber("0.65473+0.75988i"), new CNumber("0.9976+0.5804i"), new CNumber("0.33559+0.52303i"), new CNumber("0.50956+0.25486i"), new CNumber("0.00447+0.95702i"), new CNumber("0.86194+0.93487i"), new CNumber("0.68812+0.05045i"), new CNumber("0.53772+0.67041i"), new CNumber("0.60715+0.62807i"), new CNumber("0.19197+0.41375i"), new CNumber("0.87871+0.64813i"), new CNumber("0.92694+0.65877i"), new CNumber("0.20162+0.22139i"), new CNumber("0.50776+0.61336i"), new CNumber("0.17493+0.76441i"), new CNumber("0.07095+0.44298i"), new CNumber("0.20028+0.97313i"), new CNumber("0.56565+0.3442i"), new CNumber("0.54865+0.1373i")},
                {new CNumber("0.35836+0.10142i"), new CNumber("0.42694+0.08581i"), new CNumber("0.96415+0.873i"), new CNumber("0.53311+0.13981i"), new CNumber("0.12046+0.92119i"), new CNumber("0.14276+0.2659i"), new CNumber("0.3735+0.51876i"), new CNumber("0.02133+0.18251i"), new CNumber("0.91626+0.06119i"), new CNumber("0.23188+0.4063i"), new CNumber("0.87647+0.9662i"), new CNumber("0.92806+0.75166i"), new CNumber("0.19465+0.44349i"), new CNumber("0.55805+0.85738i"), new CNumber("0.87049+0.45549i"), new CNumber("0.70528+0.35606i"), new CNumber("0.7041+0.85725i"), new CNumber("0.35881+0.62024i"), new CNumber("0.86947+0.8897i"), new CNumber("0.32369+0.88844i"), new CNumber("0.01842+0.70359i"), new CNumber("0.07599+0.32326i"), new CNumber("0.27632+0.33874i")},
                {new CNumber("0.37254+0.19227i"), new CNumber("0.67085+0.85876i"), new CNumber("0.14158+0.68015i"), new CNumber("0.66505+0.97763i"), new CNumber("0.27432+0.11295i"), new CNumber("0.81077+0.54127i"), new CNumber("0.27238+0.46291i"), new CNumber("0.67764+0.30969i"), new CNumber("0.99248+0.43405i"), new CNumber("0.00531+0.15442i"), new CNumber("0.89572+0.07097i"), new CNumber("0.37489+0.08476i"), new CNumber("0.62109+0.66296i"), new CNumber("0.22484+0.12581i"), new CNumber("0.14924+0.90619i"), new CNumber("0.79776+0.89686i"), new CNumber("0.95534+0.57681i"), new CNumber("0.9634+0.51743i"), new CNumber("0.04747+0.75849i"), new CNumber("0.82984+0.25196i"), new CNumber("0.79092+0.26543i"), new CNumber("0.73121+0.65118i"), new CNumber("0.33196+0.19495i")},
                {new CNumber("0.81364+0.65333i"), new CNumber("0.96073+0.11999i"), new CNumber("0.03582+0.0292i"), new CNumber("0.14991+0.92675i"), new CNumber("0.95956+0.74278i"), new CNumber("0.54969+0.80374i"), new CNumber("0.81367+0.80792i"), new CNumber("0.03201+0.88376i"), new CNumber("0.44266+0.50854i"), new CNumber("0.36748+0.80881i"), new CNumber("0.36295+0.38929i"), new CNumber("0.18589+0.28266i"), new CNumber("0.98489+0.36438i"), new CNumber("0.41532+0.45599i"), new CNumber("0.58643+0.79833i"), new CNumber("0.87618+0.34262i"), new CNumber("0.81122+0.88096i"), new CNumber("0.52657+0.0181i"), new CNumber("0.71431+0.98372i"), new CNumber("0.87375+0.97527i"), new CNumber("0.73537+0.78923i"), new CNumber("0.8043+0.46386i"), new CNumber("0.28914+0.22724i")},
                {new CNumber("0.70491+0.70188i"), new CNumber("0.51804+0.81427i"), new CNumber("0.39958+0.19758i"), new CNumber("0.58493+0.72366i"), new CNumber("0.8421+0.12674i"), new CNumber("0.28403+0.83371i"), new CNumber("0.04987+0.53832i"), new CNumber("0.71175+0.58354i"), new CNumber("0.39884+0.89792i"), new CNumber("0.35081+0.59574i"), new CNumber("0.93969+0.1087i"), new CNumber("0.26477+0.77209i"), new CNumber("0.85254+0.09502i"), new CNumber("0.60926+0.47856i"), new CNumber("0.62183+0.98267i"), new CNumber("0.50454+0.33592i"), new CNumber("0.69385+0.6194i"), new CNumber("0.46615+0.45378i"), new CNumber("0.92443+0.52533i"), new CNumber("0.00837+0.9444i"), new CNumber("0.61049+0.14338i"), new CNumber("0.16185+0.44521i"), new CNumber("0.68142+0.62116i")},
                {new CNumber("0.46507+0.89727i"), new CNumber("0.36422+0.59032i"), new CNumber("0.1536+0.6329i"), new CNumber("0.04184+0.55612i"), new CNumber("0.26017+0.69282i"), new CNumber("0.20335+0.00405i"), new CNumber("0.42897+0.40669i"), new CNumber("0.07725+0.34163i"), new CNumber("0.99116+0.15044i"), new CNumber("0.89902+0.90403i"), new CNumber("0.50925+0.03437i"), new CNumber("0.88877+0.19088i"), new CNumber("0.31863+0.27534i"), new CNumber("0.48934+0.57884i"), new CNumber("0.52702+0.84194i"), new CNumber("0.18837+0.52471i"), new CNumber("0.40617+0.61633i"), new CNumber("0.81464+0.11444i"), new CNumber("0.63033+0.20525i"), new CNumber("0.53453+0.03308i"), new CNumber("0.88066+0.4585i"), new CNumber("0.07076+0.58566i"), new CNumber("0.98522+0.69849i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber(-0.3060695364, 0.3316025788), new CNumber(0.348758174, 0.700295456), new CNumber(0.1507853792, 0.9095346376), new CNumber(-0.34027467359999997, 0.4695810112), new CNumber(-0.442902422, 0.7609605079999999), new CNumber(-0.085869914, 0.247947664), new CNumber(-0.364602918, 0.9011378159999999), new CNumber(-0.04348780280000003, 0.9769440576), new CNumber(-0.3379530512, 0.9935104184), new CNumber(-0.18585905200000002, 0.793562134), new CNumber(-0.30406894440000004, 0.5708884148)},
                {new CNumber(0.17332276959999998, 0.2129078038), new CNumber(0.13877389889999997, 0.5480962997), new CNumber(-0.0014265840000000085, 0.43013571399999995), new CNumber(0.12993304849999998, 0.2598158195), new CNumber(0.028629182399999994, 0.2123833492), new CNumber(-0.0914247261, 0.25089725669999996), new CNumber(-0.0890388517, 0.5557431978999999), new CNumber(0.3585232484, 0.3713964332), new CNumber(0.17183480159999998, 0.7602298097999999), new CNumber(-0.2692120317, 0.49350309389999997), new CNumber(0.26836797109999994, 0.1720385743)},
                {new CNumber(0.29820956750000005, 0.4888649273), new CNumber(-0.01951868920000001, 0.4003778796), new CNumber(-0.013164186500000008, 0.9733302099), new CNumber(-0.08648401940000003, 0.9237239844), new CNumber(0.4071776207, 1.1028241561), new CNumber(0.2144187803, 0.8210862644), new CNumber(0.3528725557, 0.2319987662), new CNumber(0.29195035119999996, 1.0156390795), new CNumber(0.8205685882999999, 0.4563085259), new CNumber(0.17064644589999994, 0.9202112832), new CNumber(0.11351172639999998, 0.6630109065999998)},
                {new CNumber(0.10103110289999997, 0.4917255307), new CNumber(0.08763098179999995, 0.6810128136), new CNumber(0.17358599630000002, 0.5542225863), new CNumber(0.38469019520000003, 0.5211405816), new CNumber(-0.0514083662, 0.9112391529999999), new CNumber(0.19817928499999998, 0.3938044698), new CNumber(-0.0711965152, 0.1230761726), new CNumber(0.2770738185, 0.5055670977), new CNumber(-0.3889630555, 0.5629949963), new CNumber(0.17155163890000003, 0.6912124591), new CNumber(-0.11011721919999999, 0.24764475019999999)},
                {new CNumber(-0.2768803297, 0.4739500342), new CNumber(-0.10435888690000003, 1.1126055192000002), new CNumber(0.29177823299999994, 1.8237672432000003), new CNumber(0.13112682159999994, 0.9289735361), new CNumber(-0.5046259278, 1.3350512718999998), new CNumber(-0.2618208784, 0.4217147562), new CNumber(-0.2892679765, 2.3070579665), new CNumber(-0.6677849730000001, 1.5844368502), new CNumber(-0.4905643675999999, 0.8226165196), new CNumber(-0.10651973740000004, 1.1454259916999998), new CNumber(-0.7494584997000002, 1.1494149167)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.03074601229999996, 0.38662297), new CNumber(-0.4481117106, 0.9387060263), new CNumber(0.04731514739999999, 0.7330740530000001), new CNumber(-0.3971851536999999, 0.9776822462), new CNumber(-0.5167736116, 1.0251282487), new CNumber(-0.4581868534, 0.8309926957), new CNumber(-0.8003288335000001, 0.3751776325), new CNumber(0.013017728700000009, 0.41059236660000004), new CNumber(-0.5454624232, 1.097233165), new CNumber(0.09995373940000002, 1.1853524856), new CNumber(-0.3580233945, 0.780290392)},
                {new CNumber(-0.1598398829999999, 0.7575118317999999), new CNumber(-0.13832342699999997, 0.6590241368), new CNumber(-0.11649829439999995, 1.232114131), new CNumber(-0.2467757028, 0.3157613952), new CNumber(0.04933067580000003, 0.9974883483999999), new CNumber(-0.5143208578, 0.8939020416), new CNumber(-0.6581470011999999, 0.5746908802), new CNumber(0.00572493040000005, 0.9985417322), new CNumber(-0.08307246399999998, 0.7830736548), new CNumber(-0.15513695160000002, 0.8648640731999999), new CNumber(-0.019989464199999973, 0.7307557444)},
                {new CNumber(-0.14544583980000003, 0.9912384508000001), new CNumber(-0.44872586560000005, 0.9579569728), new CNumber(0.22278971549999982, 2.9111484078), new CNumber(-0.1399135259, 2.0655533996999997), new CNumber(0.7762645653, 2.530124631), new CNumber(-0.7555320964000001, 2.7135990291), new CNumber(-0.7863866324, 1.6606081466), new CNumber(0.41902252799999995, 2.3268939181999997), new CNumber(0.16321233989999995, 2.8717600535), new CNumber(0.6212351062999999, 1.7220599834000003), new CNumber(-0.11687056620000014, 1.9238829277)},
                {new CNumber(0.060536667700000046, 0.6518694048999999), new CNumber(0.08048906310000004, 0.5418739860999999), new CNumber(0.002529656700000049, 0.9176109295), new CNumber(-0.23122435610000003, 0.14024318930000002), new CNumber(0.34112388300000007, 0.6638082576), new CNumber(-0.12452363239999997, 0.6454756754000001), new CNumber(-0.34714446359999995, 0.8030230084000001), new CNumber(0.2745868564, 0.7486142447999999), new CNumber(0.12349327780000002, 0.44496031399999997), new CNumber(-0.06025278769999998, 0.7325064317000001), new CNumber(0.19259437580000008, 0.6521394146)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)}
        };
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.50758+0.81099i"), new CNumber("0.88299+0.28232i"), new CNumber("0.47478+0.65168i"), new CNumber("0.43933+0.06796i")};
        aRowIndices = new int[]{2, 2, 3, 3};
        aColIndices = new int[]{1, 2, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.63493+0.59114i"), new CNumber("0.71629+0.05757i"), new CNumber("0.54928+0.58241i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("0.6374685399+1.2794603432i")},
                {new CNumber("0.22699317869999996+0.7684039281i")},
                {new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.81281+0.18058i"), new CNumber("0.89609+0.98022i"), new CNumber("0.31595+0.02977i"), new CNumber("0.91545+0.12474i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.63272+0.54877i"), new CNumber("0.98762+0.16713i"), new CNumber("0.86487+0.39664i")},
                {new CNumber("0.55436+0.24057i"), new CNumber("0.38471+0.70146i"), new CNumber("0.94253+0.49763i")}};
        b = new CMatrix(bEntries);

        expEntries = new CNumber[][]{
                {new CNumber("0.41518425659999997+0.5603023213i"), new CNumber("0.40714722099999995+0.2956440305i")},
                {new CNumber("0.33612181430000004+1.1941582786i"), new CNumber("0.3616115873+0.9920462341999998i")},
                {new CNumber("0.7422683679000001+0.4709879718i"), new CNumber("0.8007647223000001+0.5731265757i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.55235+0.26182i"), new CNumber("0.58564+0.70848i"), new CNumber("0.9946+0.60902i"), new CNumber("0.06743+0.94976i")};
        aRowIndices = new int[]{1, 1, 2, 3};
        aColIndices = new int[]{1, 2, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber(0.66261, 6e-05), new CNumber("0.71057+0.4734i")},
                {new CNumber("0.11202+0.18842i"), new CNumber("0.83545+0.0718i")},
                {new CNumber("0.94928+0.93052i"), new CNumber("0.82029+0.42366i")},
                {new CNumber("0.97127+0.29525i"), new CNumber("0.24912+0.17965i")},
                {new CNumber("0.36097+0.25905i"), new CNumber("0.35992+0.86852i")}};
        b = new CMatrix(bEntries);

        CooCMatrix final0a = a;
        CMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new CNumber[]{new CNumber("0.07076+0.6121i"), new CNumber("0.97426+0.34129i"), new CNumber("0.8445+0.7417i"), new CNumber("0.00984+0.91537i")};
        aRowIndices = new int[]{0, 0, 3, 4};
        aColIndices = new int[]{0, 1, 0, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[][]{
                {new CNumber("0.38125+0.24542i"), new CNumber("0.41152+0.75363i"), new CNumber("0.36885+0.89808i"), new CNumber("0.26772+0.45414i"), new CNumber("0.94109+0.27041i")},
                {new CNumber("0.30577+0.09287i"), new CNumber("0.37228+0.02103i"), new CNumber("0.78261+0.01156i"), new CNumber("0.20146+0.81422i"), new CNumber("0.06715+0.30742i")},
                {new CNumber("0.02808+0.6474i"), new CNumber("0.68792+0.35342i"), new CNumber("0.88728+0.56851i"), new CNumber("0.98438+0.63995i"), new CNumber("0.79142+0.98226i")},
                {new CNumber("0.51624+0.79869i"), new CNumber("0.14438+0.03387i"), new CNumber("0.9602+0.27432i"), new CNumber("0.89428+0.0772i"), new CNumber("0.16939+0.48586i")},
                {new CNumber("0.29733+0.71067i"), new CNumber("0.75773+0.5138i"), new CNumber("0.41635+0.36381i"), new CNumber("0.21207+0.03446i"), new CNumber("0.59491+0.14205i")}};
        b = new CMatrix(bEntries);

        CooCMatrix final1a = a;
        CMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }
}
