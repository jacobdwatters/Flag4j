package org.flag4j.sparse_tensor;

import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooTensorSpElemBinOpsTests {

    CooTensor A;
    Shape aShape;
    double[] aEntries;
    int[][] aIndices;

    CooTensor B;
    Shape bShape;
    double[] bEntries;
    int[][] bIndices;

    CooTensor exp;
    Shape expShape;
    double[] expEntries;
    int[][] expIndices;

    @Test
    void elemMultTest() {
        // ------------------------ Sub-case 1 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        aEntries = new double[]{-0.16684361210076226, -0.5052399843372454, -1.1124588299404807, -0.022660648601335445};
        aIndices = new int[][]{
                {0, 1, 0, 1},
                {0, 2, 0, 0},
                {0, 2, 0, 1},
                {1, 0, 0, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 1, 2);
        bEntries = new double[]{1.4418647728434826, 2.0726552515293313, -0.3569517918248444, 0.9715372108728183, -0.5988716850281584, 0.7384018809658556};
        bIndices = new int[][]{
                {0, 1, 0, 0},
                {0, 2, 0, 1},
                {1, 0, 0, 0},
                {1, 1, 0, 0},
                {1, 2, 0, 1},
                {2, 3, 0, 1}};
        B = new CooTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 1, 2);
        expEntries = new double[]{-2.3057436359863126, 0.00808875912215984};
        expIndices = new int[][]{
                {0, 2, 0, 1},
                {1, 0, 0, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------------ Sub-case 2 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        aEntries = new double[]{1.620948207485194, -0.6318752045782683};
        aIndices = new int[][]{
                {0, 3, 0, 1},
                {2, 1, 0, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 1, 2);
        bEntries = new double[]{0.2971890286543011};
        bIndices = new int[][]{
                {2, 2, 0, 1}};
        B = new CooTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 1, 2);
        expEntries = new double[]{};
        expIndices = new int[][]{
        };
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------------ Sub-case 3 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooTensor(aShape);

        bShape = new Shape(3, 4, 1, 2, 1);
        B = new CooTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));

        // ------------------------ Sub-case 4 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooTensor(aShape);

        bShape = new Shape(3, 12, 1, 2);
        B = new CooTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));
    }


    @Test
    void addTest() {
        // ------------------------------ sub-case 1 ------------------------------
        aShape = new Shape(3, 4, 6, 1, 5);
        aEntries = new double[]{0.21271988542678913, 0.4910515378490831, -0.24639003895647324, 0.06641088749813011, 0.16500346832399831, -0.712196883239192, -0.9027269554225115, 0.5302886660113901, 0.0554796776957832, -1.069050745498048, 1.635492834799256, -0.24852013364240885, 2.3628236961279203, -0.15743529013594634, 0.03331901921129941, -1.262779235757704, -1.722636484760481, -0.15095034333007482, -0.23551692651137207, 0.703144233399921, -0.32415584931390173, 0.4690748702304838, 0.5878363790342139, 1.5120735176478024, -0.5363142790993337, 1.1089772351757516, -1.2364658717672565, -0.12980209770875445, 1.1384712325644109, -1.9666222600660057, 0.3303069965746997, -0.3087781084499124, 0.6644884845502304, 0.25983980776189985, -1.682467890553107, -1.039639118211318, 0.980396784918533, -0.33855547201752695, 0.8130472164219463, 0.11574933668073632, 0.4876965002122167, -0.4099683710072427, 0.27379575847214693, 0.07234280566759424, 0.7415165524784838, 1.1139497969769019, -0.6981164442443558, 0.7082474745518845, 0.6928676615023016, 0.19023223795392719, -0.7004333004673201, 1.3768256284398517, 0.1468548398333117, 0.8733236418909152, 0.311778156088708, -1.1656713775090164, -0.19921128583708514, 1.1728712714013305, 0.027399683603257403, 0.6620043229328887, -0.39492787055113454, -0.5550301570279559, 0.6638581241381123, 0.10899495829142322, -0.7947932192969646, 0.8877649747228633, -2.510241251220681, 0.5594096043704835, -2.6156284691224867, 0.03836071433776447, 0.5014356687803425};
        aIndices = new int[][]{
                {0, 0, 0, 0, 1},
                {0, 0, 1, 0, 2},
                {0, 0, 1, 0, 4},
                {0, 0, 2, 0, 1},
                {0, 0, 2, 0, 3},
                {0, 0, 2, 0, 4},
                {0, 0, 3, 0, 0},
                {0, 0, 4, 0, 4},
                {0, 0, 5, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 1, 2, 0, 1},
                {0, 1, 3, 0, 0},
                {0, 1, 5, 0, 0},
                {0, 1, 5, 0, 1},
                {0, 1, 5, 0, 2},
                {0, 1, 5, 0, 3},
                {0, 2, 0, 0, 4},
                {0, 2, 2, 0, 3},
                {0, 2, 3, 0, 0},
                {0, 2, 3, 0, 2},
                {0, 2, 4, 0, 0},
                {0, 2, 4, 0, 4},
                {0, 2, 5, 0, 2},
                {0, 3, 2, 0, 2},
                {0, 3, 2, 0, 3},
                {0, 3, 2, 0, 4},
                {0, 3, 4, 0, 1},
                {0, 3, 4, 0, 3},
                {0, 3, 5, 0, 4},
                {1, 0, 0, 0, 3},
                {1, 0, 1, 0, 1},
                {1, 0, 2, 0, 2},
                {1, 0, 3, 0, 0},
                {1, 0, 4, 0, 0},
                {1, 0, 5, 0, 1},
                {1, 1, 0, 0, 4},
                {1, 1, 1, 0, 4},
                {1, 1, 3, 0, 0},
                {1, 1, 3, 0, 1},
                {1, 1, 4, 0, 0},
                {1, 2, 2, 0, 1},
                {1, 2, 4, 0, 3},
                {1, 2, 5, 0, 3},
                {1, 3, 0, 0, 3},
                {1, 3, 2, 0, 0},
                {1, 3, 3, 0, 4},
                {1, 3, 5, 0, 3},
                {1, 3, 5, 0, 4},
                {2, 0, 1, 0, 4},
                {2, 0, 3, 0, 1},
                {2, 0, 5, 0, 4},
                {2, 1, 0, 0, 2},
                {2, 1, 0, 0, 3},
                {2, 1, 2, 0, 2},
                {2, 1, 2, 0, 3},
                {2, 1, 3, 0, 1},
                {2, 1, 4, 0, 4},
                {2, 2, 2, 0, 2},
                {2, 2, 2, 0, 3},
                {2, 2, 3, 0, 2},
                {2, 2, 3, 0, 3},
                {2, 2, 4, 0, 3},
                {2, 2, 5, 0, 1},
                {2, 2, 5, 0, 3},
                {2, 3, 1, 0, 1},
                {2, 3, 1, 0, 2},
                {2, 3, 1, 0, 4},
                {2, 3, 3, 0, 0},
                {2, 3, 3, 0, 3},
                {2, 3, 5, 0, 0},
                {2, 3, 5, 0, 3}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 6, 1, 5);
        bEntries = new double[]{0.4452122047056537, -0.9497516327600265, -1.9156122460622127, -0.6996728596976195, -1.3149595445441993, -0.35720080890725536, -1.2680188421281426, -0.43664074764597705, -0.021663210187867588, 0.5080293933893716, -0.7721970069128593, -0.40397112092278037, 0.09861035580802854, 2.145762790156657, 0.24972799251636188, -1.184346275948209, 0.5332680892301638, 1.085408338164909, -0.4663489646956424, -0.47930029680790354, -0.032101578354987774, -1.467964504073855, 0.019880407810883963, 0.1092198472073466, 0.40615922039703517, -0.8907742091410573, 0.5221931312466919, -1.2449681251500704, 0.4767871798521082, 0.2847869484063784, 0.5934528169521122, -1.6018108726064841, -1.006753177962152, 0.4234447816878154, 0.572648996383347, 0.5458070374666198, 0.20751519786275413, 1.4819111731736458, -0.4215273977053324, 0.8276204939779518, -0.8970190235235809, -0.4089956174814017, 3.8447900599566767, -1.3982918189227995, 2.7497232877428877, -1.2373906111872148, 1.423393056780044, -0.8997150090269198, -0.8133241165221952, -1.0321980382185065, 0.6138808599258287, -0.3475095765449233, 1.3206130398073723, -1.052441576735374, 0.26150935284677895, -0.5175704694289492, -1.4640719753541447, -2.2976846611324517, -2.324671711561876, -0.7695128098063991, 0.48907014821949163, 1.422228853385413, -1.4924747118888395, 0.10619881860235283, -0.5745319938512158, -0.564071153032398, -0.652495907584853, -0.14382771421924054, -0.508776368080984, 1.0473388351981363, -1.5396488960146058, 1.8906564665466268, 0.7115464810691853, -0.8896885613701434, 0.8798999211414029, 0.3321858676642957, -0.1980261328270126, -0.02518325887305506, 1.1736942761981215, 0.32129422945581415, -0.3274779852727927, 0.843815572813266, 0.9483380972743869, 0.9162331442087179, -0.784511096372904, -1.1502552471011351, 0.14875876866843532, 1.0703050467008823, 0.5504978899417385, -0.5182332569532772};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1},
                {0, 0, 0, 0, 4},
                {0, 0, 1, 0, 3},
                {0, 0, 1, 0, 4},
                {0, 0, 2, 0, 1},
                {0, 0, 2, 0, 2},
                {0, 0, 2, 0, 4},
                {0, 0, 3, 0, 2},
                {0, 0, 4, 0, 4},
                {0, 0, 5, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 1, 1, 0, 1},
                {0, 1, 1, 0, 2},
                {0, 1, 1, 0, 3},
                {0, 1, 2, 0, 2},
                {0, 1, 3, 0, 0},
                {0, 1, 4, 0, 4},
                {0, 1, 5, 0, 4},
                {0, 2, 0, 0, 1},
                {0, 2, 1, 0, 0},
                {0, 2, 1, 0, 3},
                {0, 2, 2, 0, 1},
                {0, 2, 2, 0, 2},
                {0, 2, 2, 0, 4},
                {0, 2, 4, 0, 2},
                {0, 2, 4, 0, 3},
                {0, 3, 0, 0, 1},
                {0, 3, 1, 0, 2},
                {0, 3, 4, 0, 0},
                {0, 3, 4, 0, 4},
                {0, 3, 5, 0, 0},
                {1, 0, 0, 0, 0},
                {1, 0, 0, 0, 3},
                {1, 0, 1, 0, 4},
                {1, 0, 2, 0, 2},
                {1, 0, 3, 0, 0},
                {1, 0, 3, 0, 2},
                {1, 0, 3, 0, 4},
                {1, 0, 4, 0, 0},
                {1, 0, 4, 0, 1},
                {1, 1, 0, 0, 1},
                {1, 1, 0, 0, 4},
                {1, 1, 1, 0, 4},
                {1, 1, 2, 0, 0},
                {1, 1, 3, 0, 4},
                {1, 1, 5, 0, 1},
                {1, 1, 5, 0, 4},
                {1, 2, 0, 0, 4},
                {1, 2, 1, 0, 1},
                {1, 2, 2, 0, 4},
                {1, 2, 3, 0, 3},
                {1, 2, 3, 0, 4},
                {1, 2, 4, 0, 0},
                {1, 3, 1, 0, 0},
                {1, 3, 2, 0, 3},
                {1, 3, 4, 0, 1},
                {1, 3, 5, 0, 3},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 2},
                {2, 0, 1, 0, 2},
                {2, 0, 1, 0, 3},
                {2, 0, 2, 0, 2},
                {2, 0, 3, 0, 4},
                {2, 0, 4, 0, 2},
                {2, 1, 0, 0, 3},
                {2, 1, 1, 0, 0},
                {2, 1, 1, 0, 4},
                {2, 1, 2, 0, 3},
                {2, 1, 2, 0, 4},
                {2, 1, 3, 0, 3},
                {2, 1, 4, 0, 2},
                {2, 1, 4, 0, 4},
                {2, 2, 0, 0, 0},
                {2, 2, 0, 0, 4},
                {2, 2, 1, 0, 1},
                {2, 2, 1, 0, 4},
                {2, 2, 2, 0, 1},
                {2, 2, 2, 0, 3},
                {2, 2, 2, 0, 4},
                {2, 2, 4, 0, 3},
                {2, 2, 5, 0, 1},
                {2, 2, 5, 0, 3},
                {2, 2, 5, 0, 4},
                {2, 3, 0, 0, 0},
                {2, 3, 0, 0, 4},
                {2, 3, 2, 0, 1},
                {2, 3, 3, 0, 2},
                {2, 3, 3, 0, 3},
                {2, 3, 4, 0, 4}};
        B = new CooTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 6, 1, 5);
        expEntries = new double[]{0.4452122047056537, -0.7370317473332374, -1.9156122460622127, 0.4910515378490831, -0.6996728596976195, -1.5613495835006725, -0.29078992140912524, -1.2680188421281426, 0.16500346832399831, -1.1488376308851689, -0.9027269554225115, -0.021663210187867588, 1.0383180594007617, -0.7167173292170761, -1.4730218664208283, 0.09861035580802854, 2.145762790156657, 0.24972799251636188, 1.635492834799256, -1.184346275948209, 0.2847479555877549, 1.085408338164909, 2.3628236961279203, -0.15743529013594634, 0.03331901921129941, -1.262779235757704, -0.4663489646956424, -0.47930029680790354, -1.722636484760481, -0.032101578354987774, -1.467964504073855, 0.019880407810883963, 0.1092198472073466, -0.15095034333007482, 0.40615922039703517, -0.23551692651137207, 0.703144233399921, -0.32415584931390173, -0.8907742091410573, 0.5221931312466919, 0.4690748702304838, 0.5878363790342139, -1.2449681251500704, 0.4767871798521082, 1.5120735176478024, -0.5363142790993337, 1.1089772351757516, 0.2847869484063784, -1.2364658717672565, -0.12980209770875445, 0.5934528169521122, -1.6018108726064841, 1.1384712325644109, -1.006753177962152, -1.5431774783781904, 0.3303069965746997, 0.572648996383347, 0.23702892901670747, 0.8720036824129844, 1.4819111731736458, -0.4215273977053324, 1.0874603017398516, -0.8970190235235809, -1.682467890553107, -0.4089956174814017, 2.8051509417453584, -0.41789503400426653, 2.7497232877428877, -0.33855547201752695, 0.8130472164219463, -1.2373906111872148, 0.11574933668073632, 1.423393056780044, -0.8997150090269198, -0.8133241165221952, -1.0321980382185065, 0.4876965002122167, 0.6138808599258287, -0.3475095765449233, 1.3206130398073723, -1.052441576735374, -0.4099683710072427, 0.27379575847214693, 0.07234280566759424, 0.26150935284677895, 0.7415165524784838, -0.5175704694289492, 1.1139497969769019, -1.4640719753541447, -2.9958011053768074, 0.7082474745518845, -2.324671711561876, -0.7695128098063991, 0.48907014821949163, 1.422228853385413, 0.6928676615023016, -1.4924747118888395, 0.19023223795392719, 0.10619881860235283, -0.5745319938512158, -0.7004333004673201, 1.3768256284398517, -0.41721631319908625, -0.652495907584853, -0.14382771421924054, 0.8733236418909152, -0.19699821199227602, 1.0473388351981363, -1.1656713775090164, -1.5396488960146058, 1.8906564665466268, 0.5123351952321001, -0.8896885613701434, 0.8798999211414029, 0.3321858676642957, -0.1980261328270126, -0.02518325887305506, 1.1728712714013305, 1.201093959801379, 0.32129422945581415, 0.6620043229328887, -0.39492787055113454, -0.8825081423007486, 1.5076736969513784, 1.05733305556581, 0.9162331442087179, -0.784511096372904, -1.1502552471011351, -0.7947932192969646, 0.8877649747228633, -2.510241251220681, 0.14875876866843532, 0.5594096043704835, 1.0703050467008823, -2.065130579180748, -0.5182332569532772, 0.03836071433776447, 0.5014356687803425};
        expIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1},
                {0, 0, 0, 0, 4},
                {0, 0, 1, 0, 2},
                {0, 0, 1, 0, 3},
                {0, 0, 1, 0, 4},
                {0, 0, 2, 0, 1},
                {0, 0, 2, 0, 2},
                {0, 0, 2, 0, 3},
                {0, 0, 2, 0, 4},
                {0, 0, 3, 0, 0},
                {0, 0, 3, 0, 2},
                {0, 0, 4, 0, 4},
                {0, 0, 5, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 1, 1, 0, 1},
                {0, 1, 1, 0, 2},
                {0, 1, 1, 0, 3},
                {0, 1, 2, 0, 1},
                {0, 1, 2, 0, 2},
                {0, 1, 3, 0, 0},
                {0, 1, 4, 0, 4},
                {0, 1, 5, 0, 0},
                {0, 1, 5, 0, 1},
                {0, 1, 5, 0, 2},
                {0, 1, 5, 0, 3},
                {0, 1, 5, 0, 4},
                {0, 2, 0, 0, 1},
                {0, 2, 0, 0, 4},
                {0, 2, 1, 0, 0},
                {0, 2, 1, 0, 3},
                {0, 2, 2, 0, 1},
                {0, 2, 2, 0, 2},
                {0, 2, 2, 0, 3},
                {0, 2, 2, 0, 4},
                {0, 2, 3, 0, 0},
                {0, 2, 3, 0, 2},
                {0, 2, 4, 0, 0},
                {0, 2, 4, 0, 2},
                {0, 2, 4, 0, 3},
                {0, 2, 4, 0, 4},
                {0, 2, 5, 0, 2},
                {0, 3, 0, 0, 1},
                {0, 3, 1, 0, 2},
                {0, 3, 2, 0, 2},
                {0, 3, 2, 0, 3},
                {0, 3, 2, 0, 4},
                {0, 3, 4, 0, 0},
                {0, 3, 4, 0, 1},
                {0, 3, 4, 0, 3},
                {0, 3, 4, 0, 4},
                {0, 3, 5, 0, 0},
                {0, 3, 5, 0, 4},
                {1, 0, 0, 0, 0},
                {1, 0, 0, 0, 3},
                {1, 0, 1, 0, 1},
                {1, 0, 1, 0, 4},
                {1, 0, 2, 0, 2},
                {1, 0, 3, 0, 0},
                {1, 0, 3, 0, 2},
                {1, 0, 3, 0, 4},
                {1, 0, 4, 0, 0},
                {1, 0, 4, 0, 1},
                {1, 0, 5, 0, 1},
                {1, 1, 0, 0, 1},
                {1, 1, 0, 0, 4},
                {1, 1, 1, 0, 4},
                {1, 1, 2, 0, 0},
                {1, 1, 3, 0, 0},
                {1, 1, 3, 0, 1},
                {1, 1, 3, 0, 4},
                {1, 1, 4, 0, 0},
                {1, 1, 5, 0, 1},
                {1, 1, 5, 0, 4},
                {1, 2, 0, 0, 4},
                {1, 2, 1, 0, 1},
                {1, 2, 2, 0, 1},
                {1, 2, 2, 0, 4},
                {1, 2, 3, 0, 3},
                {1, 2, 3, 0, 4},
                {1, 2, 4, 0, 0},
                {1, 2, 4, 0, 3},
                {1, 2, 5, 0, 3},
                {1, 3, 0, 0, 3},
                {1, 3, 1, 0, 0},
                {1, 3, 2, 0, 0},
                {1, 3, 2, 0, 3},
                {1, 3, 3, 0, 4},
                {1, 3, 4, 0, 1},
                {1, 3, 5, 0, 3},
                {1, 3, 5, 0, 4},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 2},
                {2, 0, 1, 0, 2},
                {2, 0, 1, 0, 3},
                {2, 0, 1, 0, 4},
                {2, 0, 2, 0, 2},
                {2, 0, 3, 0, 1},
                {2, 0, 3, 0, 4},
                {2, 0, 4, 0, 2},
                {2, 0, 5, 0, 4},
                {2, 1, 0, 0, 2},
                {2, 1, 0, 0, 3},
                {2, 1, 1, 0, 0},
                {2, 1, 1, 0, 4},
                {2, 1, 2, 0, 2},
                {2, 1, 2, 0, 3},
                {2, 1, 2, 0, 4},
                {2, 1, 3, 0, 1},
                {2, 1, 3, 0, 3},
                {2, 1, 4, 0, 2},
                {2, 1, 4, 0, 4},
                {2, 2, 0, 0, 0},
                {2, 2, 0, 0, 4},
                {2, 2, 1, 0, 1},
                {2, 2, 1, 0, 4},
                {2, 2, 2, 0, 1},
                {2, 2, 2, 0, 2},
                {2, 2, 2, 0, 3},
                {2, 2, 2, 0, 4},
                {2, 2, 3, 0, 2},
                {2, 2, 3, 0, 3},
                {2, 2, 4, 0, 3},
                {2, 2, 5, 0, 1},
                {2, 2, 5, 0, 3},
                {2, 2, 5, 0, 4},
                {2, 3, 0, 0, 0},
                {2, 3, 0, 0, 4},
                {2, 3, 1, 0, 1},
                {2, 3, 1, 0, 2},
                {2, 3, 1, 0, 4},
                {2, 3, 2, 0, 1},
                {2, 3, 3, 0, 0},
                {2, 3, 3, 0, 2},
                {2, 3, 3, 0, 3},
                {2, 3, 4, 0, 4},
                {2, 3, 5, 0, 0},
                {2, 3, 5, 0, 3}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.add(B));

        // ------------------------------ sub-case 2 ------------------------------
        aShape = new Shape(3, 4, 6, 1, 5);
        aEntries = new double[]{-0.6228105703680185};
        aIndices = new int[][]{
                {2, 1, 1, 0, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 6, 1, 5);
        bEntries = new double[]{0.6237288645617417, 0.4572169708135583, -0.6223261232137111};
        bIndices = new int[][]{
                {0, 1, 5, 0, 2},
                {0, 2, 3, 0, 3},
                {0, 3, 1, 0, 3}};
        B = new CooTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 6, 1, 5);
        expEntries = new double[]{0.6237288645617417, 0.4572169708135583, -0.6223261232137111, -0.6228105703680185};
        expIndices = new int[][]{
                {0, 1, 5, 0, 2},
                {0, 2, 3, 0, 3},
                {0, 3, 1, 0, 3},
                {2, 1, 1, 0, 0}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.add(B));

        // ------------------------ Sub-case 3 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooTensor(aShape);

        bShape = new Shape(3, 4, 1, 2, 1);
        B = new CooTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.add(B));

        // ------------------------ Sub-case 4 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooTensor(aShape);

        bShape = new Shape(3, 12, 1, 2);
        B = new CooTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.add(B));
    }


    @Test
    void subTests() {
        // ------------------------ Sub-case 1 ------------------------
        aShape = new Shape(3, 4, 6, 1, 5);
        aEntries = new double[]{-0.5712449609116215, -0.49004395314295557, 1.5715008978339873, -0.21667718838818686, 0.1745431703984849, -1.1004498342966045, -0.676995637130515, -1.7320614784970159, 1.5586547564195028, 0.611153895691038, -0.11208629951030782, 0.1444101865940601, -1.4335819402326644, 0.39041521163262366, 0.8106806718394286, 1.0422855257722368, 1.4272121513673586, -0.15504834834306788, -1.448099724471449, 0.24695194385889596, 1.1330513492380099, 0.66168304985216, -0.16203755702244307, 0.8040919212753633, 0.9578074577602911, -0.5750889348437798, -1.0000201229925485, 0.4770867265534327, -1.44133976954641, -0.20458485556356368, 0.6309627756447167, -0.08803380901022782, -0.43872137489317253, -0.11012183817754316, 0.8961589435738292, -0.8456141086004313, 2.8686395931180426, -0.40191741376948503, 0.4858071825781344, -0.5943075013972965, 0.2942967344752604, -1.0826431548972546, -0.4707149930209947, 0.3952349674229234, 0.22624625058757672, -1.129416696902452, 0.004551844692751868, 0.9556659511444416, 1.358027713041654, 0.49197835259702577, -1.8175067468344335, 0.4868379358144251, -0.3410350665089444, 0.716654948250773};
        aIndices = new int[][]{
                {0, 0, 0, 0, 4},
                {0, 0, 1, 0, 0},
                {0, 0, 1, 0, 1},
                {0, 0, 1, 0, 3},
                {0, 0, 2, 0, 0},
                {0, 0, 3, 0, 2},
                {0, 0, 4, 0, 0},
                {0, 0, 4, 0, 3},
                {0, 0, 4, 0, 4},
                {0, 0, 5, 0, 0},
                {0, 0, 5, 0, 4},
                {0, 1, 2, 0, 1},
                {0, 1, 2, 0, 4},
                {0, 1, 3, 0, 3},
                {0, 1, 3, 0, 4},
                {0, 1, 4, 0, 0},
                {0, 1, 4, 0, 1},
                {0, 1, 4, 0, 4},
                {0, 2, 0, 0, 4},
                {0, 2, 1, 0, 2},
                {0, 2, 4, 0, 3},
                {0, 3, 0, 0, 4},
                {0, 3, 1, 0, 0},
                {0, 3, 2, 0, 1},
                {0, 3, 2, 0, 4},
                {0, 3, 3, 0, 2},
                {0, 3, 4, 0, 4},
                {1, 0, 0, 0, 2},
                {1, 0, 3, 0, 3},
                {1, 0, 4, 0, 4},
                {1, 0, 5, 0, 1},
                {1, 1, 5, 0, 4},
                {1, 2, 3, 0, 3},
                {1, 2, 4, 0, 0},
                {1, 3, 3, 0, 2},
                {1, 3, 4, 0, 3},
                {1, 3, 5, 0, 2},
                {2, 0, 0, 0, 1},
                {2, 0, 0, 0, 2},
                {2, 0, 2, 0, 2},
                {2, 0, 4, 0, 2},
                {2, 1, 0, 0, 0},
                {2, 1, 0, 0, 3},
                {2, 1, 1, 0, 0},
                {2, 1, 2, 0, 2},
                {2, 1, 3, 0, 1},
                {2, 1, 5, 0, 1},
                {2, 1, 5, 0, 4},
                {2, 2, 0, 0, 2},
                {2, 2, 4, 0, 4},
                {2, 3, 0, 0, 3},
                {2, 3, 2, 0, 3},
                {2, 3, 3, 0, 4},
                {2, 3, 4, 0, 4}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 6, 1, 5);
        bEntries = new double[]{0.25749787460040247, 0.42810436253612655, -1.6990399554621591, -0.14162670591726237, -1.2156389247224306, 0.7655794435655864, -1.3283848630735486, -0.9709125642022862, 0.5032372168752851, 0.35754224417321595, 0.6448918786545342, 0.6919003333683309, 0.37318821706569305, -0.4799224390852263, -0.738181938577228, -0.7196554105181794, 1.685298290445505, 0.1041052956087065, -1.9255068249497573, -1.2222243573789249, -2.8639122810420745, -0.18810275714606653, -1.135743429612358, -0.35335118028050827, -0.6288242667416721, 0.678417186142117, 1.2234810535724336, 1.3348632716714322, -0.007293378755596761, -1.0260978828600755, 0.5317952637994143, 0.3802663183093493, -1.1814694383625668, 1.778845538477331, 1.31760559540134};
        bIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 0, 2, 0, 2},
                {0, 0, 3, 0, 3},
                {0, 0, 4, 0, 4},
                {0, 1, 0, 0, 3},
                {0, 1, 5, 0, 2},
                {0, 3, 2, 0, 0},
                {0, 3, 5, 0, 3},
                {1, 0, 3, 0, 3},
                {1, 0, 5, 0, 4},
                {1, 1, 3, 0, 1},
                {1, 1, 3, 0, 3},
                {1, 3, 1, 0, 2},
                {1, 3, 2, 0, 3},
                {1, 3, 5, 0, 1},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 4},
                {2, 0, 1, 0, 3},
                {2, 0, 1, 0, 4},
                {2, 1, 2, 0, 0},
                {2, 1, 2, 0, 1},
                {2, 1, 2, 0, 2},
                {2, 1, 5, 0, 2},
                {2, 2, 0, 0, 1},
                {2, 2, 3, 0, 0},
                {2, 2, 4, 0, 4},
                {2, 2, 5, 0, 3},
                {2, 3, 0, 0, 4},
                {2, 3, 1, 0, 0},
                {2, 3, 2, 0, 0},
                {2, 3, 3, 0, 0},
                {2, 3, 3, 0, 3},
                {2, 3, 4, 0, 0},
                {2, 3, 4, 0, 2},
                {2, 3, 5, 0, 2}};
        B = new CooTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 6, 1, 5);
        expEntries = new double[]{-0.25749787460040247, -0.5712449609116215, -0.49004395314295557, 1.5715008978339873, -0.21667718838818686, 0.1745431703984849, -0.42810436253612655, -1.1004498342966045, 1.6990399554621591, -0.676995637130515, -1.7320614784970159, 1.7002814623367652, 0.611153895691038, -0.11208629951030782, 1.2156389247224306, 0.1444101865940601, -1.4335819402326644, 0.39041521163262366, 0.8106806718394286, 1.0422855257722368, 1.4272121513673586, -0.15504834834306788, -0.7655794435655864, -1.448099724471449, 0.24695194385889596, 1.1330513492380099, 0.66168304985216, -0.16203755702244307, 1.3283848630735486, 0.8040919212753633, 0.9578074577602911, -0.5750889348437798, -1.0000201229925485, 0.9709125642022862, 0.4770867265534327, -1.944576986421695, -0.20458485556356368, 0.6309627756447167, -0.35754224417321595, -0.6448918786545342, -0.6919003333683309, -0.08803380901022782, -0.43872137489317253, -0.11012183817754316, -0.37318821706569305, 0.4799224390852263, 0.8961589435738292, -0.8456141086004313, 0.738181938577228, 2.8686395931180426, 0.7196554105181794, -0.40191741376948503, 0.4858071825781344, -1.685298290445505, -0.1041052956087065, 1.9255068249497573, -0.5943075013972965, 0.2942967344752604, -1.0826431548972546, -0.4707149930209947, 0.3952349674229234, 1.2222243573789249, 2.8639122810420745, 0.41434900773364325, -1.129416696902452, 0.004551844692751868, 1.135743429612358, 0.9556659511444416, 0.35335118028050827, 1.358027713041654, 0.6288242667416721, -0.18643883354509122, -1.2234810535724336, -1.8175067468344335, -1.3348632716714322, 0.007293378755596761, 1.0260978828600755, 0.4868379358144251, -0.5317952637994143, -0.3802663183093493, -0.3410350665089444, 1.1814694383625668, -1.778845538477331, 0.716654948250773, -1.31760559540134};
        expIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 4},
                {0, 0, 1, 0, 0},
                {0, 0, 1, 0, 1},
                {0, 0, 1, 0, 3},
                {0, 0, 2, 0, 0},
                {0, 0, 2, 0, 2},
                {0, 0, 3, 0, 2},
                {0, 0, 3, 0, 3},
                {0, 0, 4, 0, 0},
                {0, 0, 4, 0, 3},
                {0, 0, 4, 0, 4},
                {0, 0, 5, 0, 0},
                {0, 0, 5, 0, 4},
                {0, 1, 0, 0, 3},
                {0, 1, 2, 0, 1},
                {0, 1, 2, 0, 4},
                {0, 1, 3, 0, 3},
                {0, 1, 3, 0, 4},
                {0, 1, 4, 0, 0},
                {0, 1, 4, 0, 1},
                {0, 1, 4, 0, 4},
                {0, 1, 5, 0, 2},
                {0, 2, 0, 0, 4},
                {0, 2, 1, 0, 2},
                {0, 2, 4, 0, 3},
                {0, 3, 0, 0, 4},
                {0, 3, 1, 0, 0},
                {0, 3, 2, 0, 0},
                {0, 3, 2, 0, 1},
                {0, 3, 2, 0, 4},
                {0, 3, 3, 0, 2},
                {0, 3, 4, 0, 4},
                {0, 3, 5, 0, 3},
                {1, 0, 0, 0, 2},
                {1, 0, 3, 0, 3},
                {1, 0, 4, 0, 4},
                {1, 0, 5, 0, 1},
                {1, 0, 5, 0, 4},
                {1, 1, 3, 0, 1},
                {1, 1, 3, 0, 3},
                {1, 1, 5, 0, 4},
                {1, 2, 3, 0, 3},
                {1, 2, 4, 0, 0},
                {1, 3, 1, 0, 2},
                {1, 3, 2, 0, 3},
                {1, 3, 3, 0, 2},
                {1, 3, 4, 0, 3},
                {1, 3, 5, 0, 1},
                {1, 3, 5, 0, 2},
                {2, 0, 0, 0, 0},
                {2, 0, 0, 0, 1},
                {2, 0, 0, 0, 2},
                {2, 0, 0, 0, 4},
                {2, 0, 1, 0, 3},
                {2, 0, 1, 0, 4},
                {2, 0, 2, 0, 2},
                {2, 0, 4, 0, 2},
                {2, 1, 0, 0, 0},
                {2, 1, 0, 0, 3},
                {2, 1, 1, 0, 0},
                {2, 1, 2, 0, 0},
                {2, 1, 2, 0, 1},
                {2, 1, 2, 0, 2},
                {2, 1, 3, 0, 1},
                {2, 1, 5, 0, 1},
                {2, 1, 5, 0, 2},
                {2, 1, 5, 0, 4},
                {2, 2, 0, 0, 1},
                {2, 2, 0, 0, 2},
                {2, 2, 3, 0, 0},
                {2, 2, 4, 0, 4},
                {2, 2, 5, 0, 3},
                {2, 3, 0, 0, 3},
                {2, 3, 0, 0, 4},
                {2, 3, 1, 0, 0},
                {2, 3, 2, 0, 0},
                {2, 3, 2, 0, 3},
                {2, 3, 3, 0, 0},
                {2, 3, 3, 0, 3},
                {2, 3, 3, 0, 4},
                {2, 3, 4, 0, 0},
                {2, 3, 4, 0, 2},
                {2, 3, 4, 0, 4},
                {2, 3, 5, 0, 2}};
        exp = new CooTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.sub(B));

        // ------------------------ Sub-case 2 ------------------------
        aShape = new Shape(3, 4, 6, 1, 5);
        aEntries = new double[]{-1.083066215874416};
        aIndices = new int[][]{
                {2, 3, 0, 0, 4}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 6, 1, 5);
        bEntries = new double[]{0.4701746769877878, -1.0749912650738502, 1.1109847828371984};
        bIndices = new int[][]{
                {1, 3, 1, 0, 4},
                {2, 0, 3, 0, 4},
                {2, 2, 3, 0, 3}};
        B = new CooTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 6, 1, 5);
        expEntries = new double[]{-0.4701746769877878, 1.0749912650738502, -1.1109847828371984, -1.083066215874416};
        expIndices = new int[][]{
                {1, 3, 1, 0, 4},
                {2, 0, 3, 0, 4},
                {2, 2, 3, 0, 3},
                {2, 3, 0, 0, 4}};
        exp = new CooTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.sub(B));

        // ------------------------ Sub-case 3 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooTensor(aShape);

        bShape = new Shape(3, 4, 1, 2, 1);
        B = new CooTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.sub(B));

        // ------------------------ Sub-case 4 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooTensor(aShape);

        bShape = new Shape(3, 12, 1, 2);
        B = new CooTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
    }
}
