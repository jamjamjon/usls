use crate::Color;

pub enum ColorMap256 {
    Turbo,
    Inferno,
    Plasma,
    Viridis,
    Magma,
    BentCoolWarm,
    BlackBody,
    ExtendedKindLmann,
    KindLmann,
    SmoothCoolWarm,
}

impl From<&str> for ColorMap256 {
    fn from(s: &str) -> Self {
        match s {
            "turbo" | "Turbo" | "TURBO" => Self::Turbo,
            "inferno" | "Inferno" | "INFERNO" => Self::Inferno,
            "plasma" | "Plasma" | "PLASMA" => Self::Plasma,
            "viridis" | "Viridis" | "VIRIDIS" => Self::Viridis,
            "magma" | "Magma" | "MAGMA" => Self::Magma,
            "bentcoolwarm" | "BentCoolWarm" | "BENTCOOLWARM" => Self::BentCoolWarm,
            "blackbody" | "BlackBody" | "BLACKBODY" => Self::BlackBody,
            "extendedkindLmann" | "ExtendedKindLmann" | "EXTENDEDKINDLMANN" => {
                Self::ExtendedKindLmann
            }
            "kindlmann" | "KindLmann" | "KINDLMANN" => Self::KindLmann,
            "smoothcoolwarm" | "SmoothCoolWarm" | "SMOOTHCOOLWARM" => Self::SmoothCoolWarm,
            _ => todo!(),
        }
    }
}

impl ColorMap256 {
    pub fn data(&self) -> [Color; 256] {
        let xs = match self {
            Self::Turbo => [
                0x30123bff, 0x321543ff, 0x33184aff, 0x341b51ff, 0x351e58ff, 0x36215fff, 0x372466ff,
                0x38276dff, 0x392a73ff, 0x3a2d79ff, 0x3b2f80ff, 0x3c3286ff, 0x3d358bff, 0x3e3891ff,
                0x3f3b97ff, 0x3f3e9cff, 0x4040a2ff, 0x4143a7ff, 0x4146acff, 0x4249b1ff, 0x424bb5ff,
                0x434ebaff, 0x4451bfff, 0x4454c3ff, 0x4456c7ff, 0x4559cbff, 0x455ccfff, 0x455ed3ff,
                0x4661d6ff, 0x4664daff, 0x4666ddff, 0x4669e0ff, 0x466be3ff, 0x476ee6ff, 0x4771e9ff,
                0x4773ebff, 0x4776eeff, 0x4778f0ff, 0x477bf2ff, 0x467df4ff, 0x4680f6ff, 0x4682f8ff,
                0x4685faff, 0x4687fbff, 0x458afcff, 0x458cfdff, 0x448ffeff, 0x4391feff, 0x4294ffff,
                0x4196ffff, 0x4099ffff, 0x3e9bfeff, 0x3d9efeff, 0x3ba0fdff, 0x3aa3fcff, 0x38a5fbff,
                0x37a8faff, 0x35abf8ff, 0x33adf7ff, 0x31aff5ff, 0x2fb2f4ff, 0x2eb4f2ff, 0x2cb7f0ff,
                0x2ab9eeff, 0x28bcebff, 0x27bee9ff, 0x25c0e7ff, 0x23c3e4ff, 0x22c5e2ff, 0x20c7dfff,
                0x1fc9ddff, 0x1ecbdaff, 0x1ccdd8ff, 0x1bd0d5ff, 0x1ad2d2ff, 0x1ad4d0ff, 0x19d5cdff,
                0x18d7caff, 0x18d9c8ff, 0x18dbc5ff, 0x18ddc2ff, 0x18dec0ff, 0x18e0bdff, 0x19e2bbff,
                0x19e3b9ff, 0x1ae4b6ff, 0x1ce6b4ff, 0x1de7b2ff, 0x1fe9afff, 0x20eaacff, 0x22ebaaff,
                0x25eca7ff, 0x27eea4ff, 0x2aefa1ff, 0x2cf09eff, 0x2ff19bff, 0x32f298ff, 0x35f394ff,
                0x38f491ff, 0x3cf58eff, 0x3ff68aff, 0x43f787ff, 0x46f884ff, 0x4af880ff, 0x4ef97dff,
                0x52fa7aff, 0x55fa76ff, 0x59fb73ff, 0x5dfc6fff, 0x61fc6cff, 0x65fd69ff, 0x69fd66ff,
                0x6dfe62ff, 0x71fe5fff, 0x75fe5cff, 0x79fe59ff, 0x7dff56ff, 0x80ff53ff, 0x84ff51ff,
                0x88ff4eff, 0x8bff4bff, 0x8fff49ff, 0x92ff47ff, 0x96fe44ff, 0x99fe42ff, 0x9cfe40ff,
                0x9ffd3fff, 0xa1fd3dff, 0xa4fc3cff, 0xa7fc3aff, 0xa9fb39ff, 0xacfb38ff, 0xaffa37ff,
                0xb1f936ff, 0xb4f836ff, 0xb7f735ff, 0xb9f635ff, 0xbcf534ff, 0xbef434ff, 0xc1f334ff,
                0xc3f134ff, 0xc6f034ff, 0xc8ef34ff, 0xcbed34ff, 0xcdec34ff, 0xd0ea34ff, 0xd2e935ff,
                0xd4e735ff, 0xd7e535ff, 0xd9e436ff, 0xdbe236ff, 0xdde037ff, 0xdfdf37ff, 0xe1dd37ff,
                0xe3db38ff, 0xe5d938ff, 0xe7d739ff, 0xe9d539ff, 0xebd339ff, 0xecd13aff, 0xeecf3aff,
                0xefcd3aff, 0xf1cb3aff, 0xf2c93aff, 0xf4c73aff, 0xf5c53aff, 0xf6c33aff, 0xf7c13aff,
                0xf8be39ff, 0xf9bc39ff, 0xfaba39ff, 0xfbb838ff, 0xfbb637ff, 0xfcb336ff, 0xfcb136ff,
                0xfdae35ff, 0xfdac34ff, 0xfea933ff, 0xfea732ff, 0xfea431ff, 0xfea130ff, 0xfe9e2fff,
                0xfe9b2dff, 0xfe992cff, 0xfe962bff, 0xfe932aff, 0xfe9029ff, 0xfd8d27ff, 0xfd8a26ff,
                0xfc8725ff, 0xfc8423ff, 0xfb8122ff, 0xfb7e21ff, 0xfa7b1fff, 0xf9781eff, 0xf9751dff,
                0xf8721cff, 0xf76f1aff, 0xf66c19ff, 0xf56918ff, 0xf46617ff, 0xf36315ff, 0xf26014ff,
                0xf15d13ff, 0xf05b12ff, 0xef5811ff, 0xed5510ff, 0xec530fff, 0xeb500eff, 0xea4e0dff,
                0xe84b0cff, 0xe7490cff, 0xe5470bff, 0xe4450aff, 0xe2430aff, 0xe14109ff, 0xdf3f08ff,
                0xdd3d08ff, 0xdc3b07ff, 0xda3907ff, 0xd83706ff, 0xd63506ff, 0xd43305ff, 0xd23105ff,
                0xd02f05ff, 0xce2d04ff, 0xcc2b04ff, 0xca2a04ff, 0xc82803ff, 0xc52603ff, 0xc32503ff,
                0xc12302ff, 0xbe2102ff, 0xbc2002ff, 0xb91e02ff, 0xb71d02ff, 0xb41b01ff, 0xb21a01ff,
                0xaf1801ff, 0xac1701ff, 0xa91601ff, 0xa71401ff, 0xa41301ff, 0xa11201ff, 0x9e1001ff,
                0x9b0f01ff, 0x980e01ff, 0x950d01ff, 0x920b01ff, 0x8e0a01ff, 0x8b0902ff, 0x880802ff,
                0x850702ff, 0x810602ff, 0x7e0502ff, 0x7a0403ff,
            ],
            Self::Inferno => [
                0x000004ff, 0x010005ff, 0x010106ff, 0x010108ff, 0x02010aff, 0x02020cff, 0x02020eff,
                0x030210ff, 0x040312ff, 0x040314ff, 0x050417ff, 0x060419ff, 0x07051bff, 0x08051dff,
                0x09061fff, 0x0a0722ff, 0x0b0724ff, 0x0c0826ff, 0x0d0829ff, 0x0e092bff, 0x10092dff,
                0x110a30ff, 0x120a32ff, 0x140b34ff, 0x150b37ff, 0x160b39ff, 0x180c3cff, 0x190c3eff,
                0x1b0c41ff, 0x1c0c43ff, 0x1e0c45ff, 0x1f0c48ff, 0x210c4aff, 0x230c4cff, 0x240c4fff,
                0x260c51ff, 0x280b53ff, 0x290b55ff, 0x2b0b57ff, 0x2d0b59ff, 0x2f0a5bff, 0x310a5cff,
                0x320a5eff, 0x340a5fff, 0x360961ff, 0x380962ff, 0x390963ff, 0x3b0964ff, 0x3d0965ff,
                0x3e0966ff, 0x400a67ff, 0x420a68ff, 0x440a68ff, 0x450a69ff, 0x470b6aff, 0x490b6aff,
                0x4a0c6bff, 0x4c0c6bff, 0x4d0d6cff, 0x4f0d6cff, 0x510e6cff, 0x520e6dff, 0x540f6dff,
                0x550f6dff, 0x57106eff, 0x59106eff, 0x5a116eff, 0x5c126eff, 0x5d126eff, 0x5f136eff,
                0x61136eff, 0x62146eff, 0x64156eff, 0x65156eff, 0x67166eff, 0x69166eff, 0x6a176eff,
                0x6c186eff, 0x6d186eff, 0x6f196eff, 0x71196eff, 0x721a6eff, 0x741a6eff, 0x751b6eff,
                0x771c6dff, 0x781c6dff, 0x7a1d6dff, 0x7c1d6dff, 0x7d1e6dff, 0x7f1e6cff, 0x801f6cff,
                0x82206cff, 0x84206bff, 0x85216bff, 0x87216bff, 0x88226aff, 0x8a226aff, 0x8c2369ff,
                0x8d2369ff, 0x8f2469ff, 0x902568ff, 0x922568ff, 0x932667ff, 0x952667ff, 0x972766ff,
                0x982766ff, 0x9a2865ff, 0x9b2964ff, 0x9d2964ff, 0x9f2a63ff, 0xa02a63ff, 0xa22b62ff,
                0xa32c61ff, 0xa52c60ff, 0xa62d60ff, 0xa82e5fff, 0xa92e5eff, 0xab2f5eff, 0xad305dff,
                0xae305cff, 0xb0315bff, 0xb1325aff, 0xb3325aff, 0xb43359ff, 0xb63458ff, 0xb73557ff,
                0xb93556ff, 0xba3655ff, 0xbc3754ff, 0xbd3853ff, 0xbf3952ff, 0xc03a51ff, 0xc13a50ff,
                0xc33b4fff, 0xc43c4eff, 0xc63d4dff, 0xc73e4cff, 0xc83f4bff, 0xca404aff, 0xcb4149ff,
                0xcc4248ff, 0xce4347ff, 0xcf4446ff, 0xd04545ff, 0xd24644ff, 0xd34743ff, 0xd44842ff,
                0xd54a41ff, 0xd74b3fff, 0xd84c3eff, 0xd94d3dff, 0xda4e3cff, 0xdb503bff, 0xdd513aff,
                0xde5238ff, 0xdf5337ff, 0xe05536ff, 0xe15635ff, 0xe25734ff, 0xe35933ff, 0xe45a31ff,
                0xe55c30ff, 0xe65d2fff, 0xe75e2eff, 0xe8602dff, 0xe9612bff, 0xea632aff, 0xeb6429ff,
                0xeb6628ff, 0xec6726ff, 0xed6925ff, 0xee6a24ff, 0xef6c23ff, 0xef6e21ff, 0xf06f20ff,
                0xf1711fff, 0xf1731dff, 0xf2741cff, 0xf3761bff, 0xf37819ff, 0xf47918ff, 0xf57b17ff,
                0xf57d15ff, 0xf67e14ff, 0xf68013ff, 0xf78212ff, 0xf78410ff, 0xf8850fff, 0xf8870eff,
                0xf8890cff, 0xf98b0bff, 0xf98c0aff, 0xf98e09ff, 0xfa9008ff, 0xfa9207ff, 0xfa9407ff,
                0xfb9606ff, 0xfb9706ff, 0xfb9906ff, 0xfb9b06ff, 0xfb9d07ff, 0xfc9f07ff, 0xfca108ff,
                0xfca309ff, 0xfca50aff, 0xfca60cff, 0xfca80dff, 0xfcaa0fff, 0xfcac11ff, 0xfcae12ff,
                0xfcb014ff, 0xfcb216ff, 0xfcb418ff, 0xfbb61aff, 0xfbb81dff, 0xfbba1fff, 0xfbbc21ff,
                0xfbbe23ff, 0xfac026ff, 0xfac228ff, 0xfac42aff, 0xfac62dff, 0xf9c72fff, 0xf9c932ff,
                0xf9cb35ff, 0xf8cd37ff, 0xf8cf3aff, 0xf7d13dff, 0xf7d340ff, 0xf6d543ff, 0xf6d746ff,
                0xf5d949ff, 0xf5db4cff, 0xf4dd4fff, 0xf4df53ff, 0xf4e156ff, 0xf3e35aff, 0xf3e55dff,
                0xf2e661ff, 0xf2e865ff, 0xf2ea69ff, 0xf1ec6dff, 0xf1ed71ff, 0xf1ef75ff, 0xf1f179ff,
                0xf2f27dff, 0xf2f482ff, 0xf3f586ff, 0xf3f68aff, 0xf4f88eff, 0xf5f992ff, 0xf6fa96ff,
                0xf8fb9aff, 0xf9fc9dff, 0xfafda1ff, 0xfcffa4ff,
            ],
            Self::Plasma => [
                0x0d0887ff, 0x100788ff, 0x130789ff, 0x16078aff, 0x19068cff, 0x1b068dff, 0x1d068eff,
                0x20068fff, 0x220690ff, 0x240691ff, 0x260591ff, 0x280592ff, 0x2a0593ff, 0x2c0594ff,
                0x2e0595ff, 0x2f0596ff, 0x310597ff, 0x330597ff, 0x350498ff, 0x370499ff, 0x38049aff,
                0x3a049aff, 0x3c049bff, 0x3e049cff, 0x3f049cff, 0x41049dff, 0x43039eff, 0x44039eff,
                0x46039fff, 0x48039fff, 0x4903a0ff, 0x4b03a1ff, 0x4c02a1ff, 0x4e02a2ff, 0x5002a2ff,
                0x5102a3ff, 0x5302a3ff, 0x5502a4ff, 0x5601a4ff, 0x5801a4ff, 0x5901a5ff, 0x5b01a5ff,
                0x5c01a6ff, 0x5e01a6ff, 0x6001a6ff, 0x6100a7ff, 0x6300a7ff, 0x6400a7ff, 0x6600a7ff,
                0x6700a8ff, 0x6900a8ff, 0x6a00a8ff, 0x6c00a8ff, 0x6e00a8ff, 0x6f00a8ff, 0x7100a8ff,
                0x7201a8ff, 0x7401a8ff, 0x7501a8ff, 0x7701a8ff, 0x7801a8ff, 0x7a02a8ff, 0x7b02a8ff,
                0x7d03a8ff, 0x7e03a8ff, 0x8004a8ff, 0x8104a7ff, 0x8305a7ff, 0x8405a7ff, 0x8606a6ff,
                0x8707a6ff, 0x8808a6ff, 0x8a09a5ff, 0x8b0aa5ff, 0x8d0ba5ff, 0x8e0ca4ff, 0x8f0da4ff,
                0x910ea3ff, 0x920fa3ff, 0x9410a2ff, 0x9511a1ff, 0x9613a1ff, 0x9814a0ff, 0x99159fff,
                0x9a169fff, 0x9c179eff, 0x9d189dff, 0x9e199dff, 0xa01a9cff, 0xa11b9bff, 0xa21d9aff,
                0xa31e9aff, 0xa51f99ff, 0xa62098ff, 0xa72197ff, 0xa82296ff, 0xaa2395ff, 0xab2494ff,
                0xac2694ff, 0xad2793ff, 0xae2892ff, 0xb02991ff, 0xb12a90ff, 0xb22b8fff, 0xb32c8eff,
                0xb42e8dff, 0xb52f8cff, 0xb6308bff, 0xb7318aff, 0xb83289ff, 0xba3388ff, 0xbb3488ff,
                0xbc3587ff, 0xbd3786ff, 0xbe3885ff, 0xbf3984ff, 0xc03a83ff, 0xc13b82ff, 0xc23c81ff,
                0xc33d80ff, 0xc43e7fff, 0xc5407eff, 0xc6417dff, 0xc7427cff, 0xc8437bff, 0xc9447aff,
                0xca457aff, 0xcb4679ff, 0xcc4778ff, 0xcc4977ff, 0xcd4a76ff, 0xce4b75ff, 0xcf4c74ff,
                0xd04d73ff, 0xd14e72ff, 0xd24f71ff, 0xd35171ff, 0xd45270ff, 0xd5536fff, 0xd5546eff,
                0xd6556dff, 0xd7566cff, 0xd8576bff, 0xd9586aff, 0xda5a6aff, 0xda5b69ff, 0xdb5c68ff,
                0xdc5d67ff, 0xdd5e66ff, 0xde5f65ff, 0xde6164ff, 0xdf6263ff, 0xe06363ff, 0xe16462ff,
                0xe26561ff, 0xe26660ff, 0xe3685fff, 0xe4695eff, 0xe56a5dff, 0xe56b5dff, 0xe66c5cff,
                0xe76e5bff, 0xe76f5aff, 0xe87059ff, 0xe97158ff, 0xe97257ff, 0xea7457ff, 0xeb7556ff,
                0xeb7655ff, 0xec7754ff, 0xed7953ff, 0xed7a52ff, 0xee7b51ff, 0xef7c51ff, 0xef7e50ff,
                0xf07f4fff, 0xf0804eff, 0xf1814dff, 0xf1834cff, 0xf2844bff, 0xf3854bff, 0xf3874aff,
                0xf48849ff, 0xf48948ff, 0xf58b47ff, 0xf58c46ff, 0xf68d45ff, 0xf68f44ff, 0xf79044ff,
                0xf79143ff, 0xf79342ff, 0xf89441ff, 0xf89540ff, 0xf9973fff, 0xf9983eff, 0xf99a3eff,
                0xfa9b3dff, 0xfa9c3cff, 0xfa9e3bff, 0xfb9f3aff, 0xfba139ff, 0xfba238ff, 0xfca338ff,
                0xfca537ff, 0xfca636ff, 0xfca835ff, 0xfca934ff, 0xfdab33ff, 0xfdac33ff, 0xfdae32ff,
                0xfdaf31ff, 0xfdb130ff, 0xfdb22fff, 0xfdb42fff, 0xfdb52eff, 0xfeb72dff, 0xfeb82cff,
                0xfeba2cff, 0xfebb2bff, 0xfebd2aff, 0xfebe2aff, 0xfec029ff, 0xfdc229ff, 0xfdc328ff,
                0xfdc527ff, 0xfdc627ff, 0xfdc827ff, 0xfdca26ff, 0xfdcb26ff, 0xfccd25ff, 0xfcce25ff,
                0xfcd025ff, 0xfcd225ff, 0xfbd324ff, 0xfbd524ff, 0xfbd724ff, 0xfad824ff, 0xfada24ff,
                0xf9dc24ff, 0xf9dd25ff, 0xf8df25ff, 0xf8e125ff, 0xf7e225ff, 0xf7e425ff, 0xf6e626ff,
                0xf6e826ff, 0xf5e926ff, 0xf5eb27ff, 0xf4ed27ff, 0xf3ee27ff, 0xf3f027ff, 0xf2f227ff,
                0xf1f426ff, 0xf1f525ff, 0xf0f724ff, 0xf0f921ff,
            ],
            Self::Viridis => [
                0x440154ff, 0x440256ff, 0x450457ff, 0x450559ff, 0x46075aff, 0x46085cff, 0x460a5dff,
                0x460b5eff, 0x470d60ff, 0x470e61ff, 0x471063ff, 0x471164ff, 0x471365ff, 0x481467ff,
                0x481668ff, 0x481769ff, 0x48186aff, 0x481a6cff, 0x481b6dff, 0x481c6eff, 0x481d6fff,
                0x481f70ff, 0x482071ff, 0x482173ff, 0x482374ff, 0x482475ff, 0x482576ff, 0x482677ff,
                0x482878ff, 0x482979ff, 0x472a7aff, 0x472c7aff, 0x472d7bff, 0x472e7cff, 0x472f7dff,
                0x46307eff, 0x46327eff, 0x46337fff, 0x463480ff, 0x453581ff, 0x453781ff, 0x453882ff,
                0x443983ff, 0x443a83ff, 0x443b84ff, 0x433d84ff, 0x433e85ff, 0x423f85ff, 0x424086ff,
                0x424186ff, 0x414287ff, 0x414487ff, 0x404588ff, 0x404688ff, 0x3f4788ff, 0x3f4889ff,
                0x3e4989ff, 0x3e4a89ff, 0x3e4c8aff, 0x3d4d8aff, 0x3d4e8aff, 0x3c4f8aff, 0x3c508bff,
                0x3b518bff, 0x3b528bff, 0x3a538bff, 0x3a548cff, 0x39558cff, 0x39568cff, 0x38588cff,
                0x38598cff, 0x375a8cff, 0x375b8dff, 0x365c8dff, 0x365d8dff, 0x355e8dff, 0x355f8dff,
                0x34608dff, 0x34618dff, 0x33628dff, 0x33638dff, 0x32648eff, 0x32658eff, 0x31668eff,
                0x31678eff, 0x31688eff, 0x30698eff, 0x306a8eff, 0x2f6b8eff, 0x2f6c8eff, 0x2e6d8eff,
                0x2e6e8eff, 0x2e6f8eff, 0x2d708eff, 0x2d718eff, 0x2c718eff, 0x2c728eff, 0x2c738eff,
                0x2b748eff, 0x2b758eff, 0x2a768eff, 0x2a778eff, 0x2a788eff, 0x29798eff, 0x297a8eff,
                0x297b8eff, 0x287c8eff, 0x287d8eff, 0x277e8eff, 0x277f8eff, 0x27808eff, 0x26818eff,
                0x26828eff, 0x26828eff, 0x25838eff, 0x25848eff, 0x25858eff, 0x24868eff, 0x24878eff,
                0x23888eff, 0x23898eff, 0x238a8dff, 0x228b8dff, 0x228c8dff, 0x228d8dff, 0x218e8dff,
                0x218f8dff, 0x21908dff, 0x21918cff, 0x20928cff, 0x20928cff, 0x20938cff, 0x1f948cff,
                0x1f958bff, 0x1f968bff, 0x1f978bff, 0x1f988bff, 0x1f998aff, 0x1f9a8aff, 0x1e9b8aff,
                0x1e9c89ff, 0x1e9d89ff, 0x1f9e89ff, 0x1f9f88ff, 0x1fa088ff, 0x1fa188ff, 0x1fa187ff,
                0x1fa287ff, 0x20a386ff, 0x20a486ff, 0x21a585ff, 0x21a685ff, 0x22a785ff, 0x22a884ff,
                0x23a983ff, 0x24aa83ff, 0x25ab82ff, 0x25ac82ff, 0x26ad81ff, 0x27ad81ff, 0x28ae80ff,
                0x29af7fff, 0x2ab07fff, 0x2cb17eff, 0x2db27dff, 0x2eb37cff, 0x2fb47cff, 0x31b57bff,
                0x32b67aff, 0x34b679ff, 0x35b779ff, 0x37b878ff, 0x38b977ff, 0x3aba76ff, 0x3bbb75ff,
                0x3dbc74ff, 0x3fbc73ff, 0x40bd72ff, 0x42be71ff, 0x44bf70ff, 0x46c06fff, 0x48c16eff,
                0x4ac16dff, 0x4cc26cff, 0x4ec36bff, 0x50c46aff, 0x52c569ff, 0x54c568ff, 0x56c667ff,
                0x58c765ff, 0x5ac864ff, 0x5cc863ff, 0x5ec962ff, 0x60ca60ff, 0x63cb5fff, 0x65cb5eff,
                0x67cc5cff, 0x69cd5bff, 0x6ccd5aff, 0x6ece58ff, 0x70cf57ff, 0x73d056ff, 0x75d054ff,
                0x77d153ff, 0x7ad151ff, 0x7cd250ff, 0x7fd34eff, 0x81d34dff, 0x84d44bff, 0x86d549ff,
                0x89d548ff, 0x8bd646ff, 0x8ed645ff, 0x90d743ff, 0x93d741ff, 0x95d840ff, 0x98d83eff,
                0x9bd93cff, 0x9dd93bff, 0xa0da39ff, 0xa2da37ff, 0xa5db36ff, 0xa8db34ff, 0xaadc32ff,
                0xaddc30ff, 0xb0dd2fff, 0xb2dd2dff, 0xb5de2bff, 0xb8de29ff, 0xbade28ff, 0xbddf26ff,
                0xc0df25ff, 0xc2df23ff, 0xc5e021ff, 0xc8e020ff, 0xcae11fff, 0xcde11dff, 0xd0e11cff,
                0xd2e21bff, 0xd5e21aff, 0xd8e219ff, 0xdae319ff, 0xdde318ff, 0xdfe318ff, 0xe2e418ff,
                0xe5e419ff, 0xe7e419ff, 0xeae51aff, 0xece51bff, 0xefe51cff, 0xf1e51dff, 0xf4e61eff,
                0xf6e620ff, 0xf8e621ff, 0xfbe723ff, 0xfde725ff,
            ],
            Self::Magma => [
                0x000004ff, 0x010005ff, 0x010106ff, 0x010108ff, 0x020109ff, 0x02020bff, 0x02020dff,
                0x03030fff, 0x030312ff, 0x040414ff, 0x050416ff, 0x060518ff, 0x06051aff, 0x07061cff,
                0x08071eff, 0x090720ff, 0x0a0822ff, 0x0b0924ff, 0x0c0926ff, 0x0d0a29ff, 0x0e0b2bff,
                0x100b2dff, 0x110c2fff, 0x120d31ff, 0x130d34ff, 0x140e36ff, 0x150e38ff, 0x160f3bff,
                0x180f3dff, 0x19103fff, 0x1a1042ff, 0x1c1044ff, 0x1d1147ff, 0x1e1149ff, 0x20114bff,
                0x21114eff, 0x221150ff, 0x241253ff, 0x251255ff, 0x271258ff, 0x29115aff, 0x2a115cff,
                0x2c115fff, 0x2d1161ff, 0x2f1163ff, 0x311165ff, 0x331067ff, 0x341069ff, 0x36106bff,
                0x38106cff, 0x390f6eff, 0x3b0f70ff, 0x3d0f71ff, 0x3f0f72ff, 0x400f74ff, 0x420f75ff,
                0x440f76ff, 0x451077ff, 0x471078ff, 0x491078ff, 0x4a1079ff, 0x4c117aff, 0x4e117bff,
                0x4f127bff, 0x51127cff, 0x52137cff, 0x54137dff, 0x56147dff, 0x57157eff, 0x59157eff,
                0x5a167eff, 0x5c167fff, 0x5d177fff, 0x5f187fff, 0x601880ff, 0x621980ff, 0x641a80ff,
                0x651a80ff, 0x671b80ff, 0x681c81ff, 0x6a1c81ff, 0x6b1d81ff, 0x6d1d81ff, 0x6e1e81ff,
                0x701f81ff, 0x721f81ff, 0x732081ff, 0x752181ff, 0x762181ff, 0x782281ff, 0x792282ff,
                0x7b2382ff, 0x7c2382ff, 0x7e2482ff, 0x802582ff, 0x812581ff, 0x832681ff, 0x842681ff,
                0x862781ff, 0x882781ff, 0x892881ff, 0x8b2981ff, 0x8c2981ff, 0x8e2a81ff, 0x902a81ff,
                0x912b81ff, 0x932b80ff, 0x942c80ff, 0x962c80ff, 0x982d80ff, 0x992d80ff, 0x9b2e7fff,
                0x9c2e7fff, 0x9e2f7fff, 0xa02f7fff, 0xa1307eff, 0xa3307eff, 0xa5317eff, 0xa6317dff,
                0xa8327dff, 0xaa337dff, 0xab337cff, 0xad347cff, 0xae347bff, 0xb0357bff, 0xb2357bff,
                0xb3367aff, 0xb5367aff, 0xb73779ff, 0xb83779ff, 0xba3878ff, 0xbc3978ff, 0xbd3977ff,
                0xbf3a77ff, 0xc03a76ff, 0xc23b75ff, 0xc43c75ff, 0xc53c74ff, 0xc73d73ff, 0xc83e73ff,
                0xca3e72ff, 0xcc3f71ff, 0xcd4071ff, 0xcf4070ff, 0xd0416fff, 0xd2426fff, 0xd3436eff,
                0xd5446dff, 0xd6456cff, 0xd8456cff, 0xd9466bff, 0xdb476aff, 0xdc4869ff, 0xde4968ff,
                0xdf4a68ff, 0xe04c67ff, 0xe24d66ff, 0xe34e65ff, 0xe44f64ff, 0xe55064ff, 0xe75263ff,
                0xe85362ff, 0xe95462ff, 0xea5661ff, 0xeb5760ff, 0xec5860ff, 0xed5a5fff, 0xee5b5eff,
                0xef5d5eff, 0xf05f5eff, 0xf1605dff, 0xf2625dff, 0xf2645cff, 0xf3655cff, 0xf4675cff,
                0xf4695cff, 0xf56b5cff, 0xf66c5cff, 0xf66e5cff, 0xf7705cff, 0xf7725cff, 0xf8745cff,
                0xf8765cff, 0xf9785dff, 0xf9795dff, 0xf97b5dff, 0xfa7d5eff, 0xfa7f5eff, 0xfa815fff,
                0xfb835fff, 0xfb8560ff, 0xfb8761ff, 0xfc8961ff, 0xfc8a62ff, 0xfc8c63ff, 0xfc8e64ff,
                0xfc9065ff, 0xfd9266ff, 0xfd9467ff, 0xfd9668ff, 0xfd9869ff, 0xfd9a6aff, 0xfd9b6bff,
                0xfe9d6cff, 0xfe9f6dff, 0xfea16eff, 0xfea36fff, 0xfea571ff, 0xfea772ff, 0xfea973ff,
                0xfeaa74ff, 0xfeac76ff, 0xfeae77ff, 0xfeb078ff, 0xfeb27aff, 0xfeb47bff, 0xfeb67cff,
                0xfeb77eff, 0xfeb97fff, 0xfebb81ff, 0xfebd82ff, 0xfebf84ff, 0xfec185ff, 0xfec287ff,
                0xfec488ff, 0xfec68aff, 0xfec88cff, 0xfeca8dff, 0xfecc8fff, 0xfecd90ff, 0xfecf92ff,
                0xfed194ff, 0xfed395ff, 0xfed597ff, 0xfed799ff, 0xfed89aff, 0xfdda9cff, 0xfddc9eff,
                0xfddea0ff, 0xfde0a1ff, 0xfde2a3ff, 0xfde3a5ff, 0xfde5a7ff, 0xfde7a9ff, 0xfde9aaff,
                0xfdebacff, 0xfcecaeff, 0xfceeb0ff, 0xfcf0b2ff, 0xfcf2b4ff, 0xfcf4b6ff, 0xfcf6b8ff,
                0xfcf7b9ff, 0xfcf9bbff, 0xfcfbbdff, 0xfcfdbfff,
            ],
            Self::BentCoolWarm => [
                0x3b4cc0ff, 0x3c4ec1ff, 0x3d4fc2ff, 0x3e50c2ff, 0x3f52c3ff, 0x4053c4ff, 0x4155c4ff,
                0x4256c5ff, 0x4357c5ff, 0x4459c6ff, 0x455ac7ff, 0x465bc7ff, 0x475dc8ff, 0x485ec8ff,
                0x495fc9ff, 0x4a61caff, 0x4b62caff, 0x4c64cbff, 0x4d65cbff, 0x4e66ccff, 0x4f68ccff,
                0x5169cdff, 0x526aceff, 0x536cceff, 0x546dcfff, 0x556ecfff, 0x5670d0ff, 0x5771d0ff,
                0x5972d1ff, 0x5a74d1ff, 0x5b75d2ff, 0x5c76d2ff, 0x5d78d3ff, 0x5f79d3ff, 0x607ad4ff,
                0x617cd4ff, 0x627dd5ff, 0x647ed5ff, 0x6580d6ff, 0x6681d6ff, 0x6782d6ff, 0x6983d7ff,
                0x6a85d7ff, 0x6b86d8ff, 0x6d87d8ff, 0x6e89d9ff, 0x6f8ad9ff, 0x718bdaff, 0x728ddaff,
                0x738edaff, 0x758fdbff, 0x7691dbff, 0x7792dcff, 0x7993dcff, 0x7a95dcff, 0x7b96ddff,
                0x7d97ddff, 0x7e98deff, 0x809adeff, 0x819bdeff, 0x829cdfff, 0x849edfff, 0x859fdfff,
                0x87a0e0ff, 0x88a2e0ff, 0x8aa3e1ff, 0x8ba4e1ff, 0x8ca5e1ff, 0x8ea7e2ff, 0x8fa8e2ff,
                0x91a9e2ff, 0x92abe3ff, 0x94ace3ff, 0x95ade3ff, 0x97afe4ff, 0x98b0e4ff, 0x9ab1e4ff,
                0x9bb2e5ff, 0x9db4e5ff, 0x9fb5e5ff, 0xa0b6e6ff, 0xa2b8e6ff, 0xa3b9e6ff, 0xa5bae6ff,
                0xa6bbe7ff, 0xa8bde7ff, 0xaabee7ff, 0xabbfe8ff, 0xadc1e8ff, 0xaec2e8ff, 0xb0c3e8ff,
                0xb2c4e9ff, 0xb3c6e9ff, 0xb5c7e9ff, 0xb7c8eaff, 0xb8caeaff, 0xbacbeaff, 0xbccceaff,
                0xbdcdebff, 0xbfcfebff, 0xc1d0ebff, 0xc2d1ecff, 0xc4d2ecff, 0xc6d4ecff, 0xc8d5ecff,
                0xc9d6edff, 0xcbd7edff, 0xcdd9edff, 0xcfdaedff, 0xd0dbeeff, 0xd2dceeff, 0xd4deeeff,
                0xd6dfeeff, 0xd7e0efff, 0xd9e1efff, 0xdbe3efff, 0xdde4efff, 0xdfe5f0ff, 0xe1e6f0ff,
                0xe2e8f0ff, 0xe4e9f0ff, 0xe6eaf1ff, 0xe8ebf1ff, 0xeaedf1ff, 0xeceef1ff, 0xeeeff2ff,
                0xeff0f2ff, 0xf1f2f2ff, 0xf2f1f1ff, 0xf2f0efff, 0xf1eeedff, 0xf1edebff, 0xf1ebe8ff,
                0xf1eae6ff, 0xf0e8e4ff, 0xf0e7e2ff, 0xf0e5e0ff, 0xefe4deff, 0xefe2dbff, 0xefe1d9ff,
                0xeedfd7ff, 0xeeded5ff, 0xeedcd3ff, 0xeddbd1ff, 0xedd9cfff, 0xedd8cdff, 0xecd6cbff,
                0xecd5c9ff, 0xecd3c7ff, 0xebd2c4ff, 0xebd0c2ff, 0xebcfc0ff, 0xeacdbeff, 0xeaccbcff,
                0xe9cabaff, 0xe9c9b8ff, 0xe9c7b6ff, 0xe8c5b4ff, 0xe8c4b3ff, 0xe8c2b1ff, 0xe7c1afff,
                0xe7bfadff, 0xe6beabff, 0xe6bca9ff, 0xe6bba7ff, 0xe5b9a5ff, 0xe5b8a3ff, 0xe4b6a1ff,
                0xe4b59fff, 0xe4b39eff, 0xe3b19cff, 0xe3b09aff, 0xe2ae98ff, 0xe2ad96ff, 0xe2ab94ff,
                0xe1aa93ff, 0xe1a891ff, 0xe0a78fff, 0xe0a58dff, 0xdfa38cff, 0xdfa28aff, 0xdfa088ff,
                0xde9f86ff, 0xde9d85ff, 0xdd9c83ff, 0xdd9a81ff, 0xdc9880ff, 0xdc977eff, 0xdb957cff,
                0xdb947bff, 0xda9279ff, 0xda9077ff, 0xd98f76ff, 0xd98d74ff, 0xd98c72ff, 0xd88a71ff,
                0xd8886fff, 0xd7876eff, 0xd7856cff, 0xd6846bff, 0xd68269ff, 0xd58068ff, 0xd47f66ff,
                0xd47d65ff, 0xd37b63ff, 0xd37a62ff, 0xd27860ff, 0xd2775fff, 0xd1755dff, 0xd1735cff,
                0xd0725aff, 0xd07059ff, 0xcf6e58ff, 0xcf6c56ff, 0xce6b55ff, 0xcd6953ff, 0xcd6752ff,
                0xcc6651ff, 0xcc644fff, 0xcb624eff, 0xcb604dff, 0xca5f4bff, 0xc95d4aff, 0xc95b49ff,
                0xc85948ff, 0xc85746ff, 0xc75645ff, 0xc65444ff, 0xc65243ff, 0xc55041ff, 0xc54e40ff,
                0xc44c3fff, 0xc34a3eff, 0xc3483dff, 0xc2463cff, 0xc1443aff, 0xc14239ff, 0xc04038ff,
                0xc03e37ff, 0xbf3c36ff, 0xbe3a35ff, 0xbe3734ff, 0xbd3533ff, 0xbc3232ff, 0xbc3031ff,
                0xbb2d30ff, 0xba2b2fff, 0xba282eff, 0xb9252dff, 0xb8222cff, 0xb81e2bff, 0xb71b2aff,
                0xb61629ff, 0xb51228ff, 0xb50c27ff, 0xb40426ff,
            ],
            Self::BlackBody => [
                0x000000ff, 0x030101ff, 0x070201ff, 0x0a0302ff, 0x0d0402ff, 0x100503ff, 0x120603ff,
                0x140704ff, 0x160804ff, 0x180905ff, 0x1a0a05ff, 0x1b0b06ff, 0x1d0b06ff, 0x1e0c07ff,
                0x200d08ff, 0x210e08ff, 0x220f09ff, 0x240f09ff, 0x25100aff, 0x26100aff, 0x28110bff,
                0x29110bff, 0x2b120cff, 0x2c120cff, 0x2e120dff, 0x2f130dff, 0x31130eff, 0x32130eff,
                0x34140fff, 0x36140fff, 0x37140fff, 0x391510ff, 0x3a1510ff, 0x3c1510ff, 0x3e1611ff,
                0x3f1611ff, 0x411611ff, 0x421712ff, 0x441712ff, 0x461712ff, 0x471813ff, 0x491813ff,
                0x4b1813ff, 0x4c1914ff, 0x4e1914ff, 0x501914ff, 0x511914ff, 0x531a15ff, 0x551a15ff,
                0x561a15ff, 0x581a15ff, 0x5a1b16ff, 0x5b1b16ff, 0x5d1b16ff, 0x5f1b16ff, 0x611c17ff,
                0x621c17ff, 0x641c17ff, 0x661c17ff, 0x681d18ff, 0x691d18ff, 0x6b1d18ff, 0x6d1d18ff,
                0x6f1d19ff, 0x701e19ff, 0x721e19ff, 0x741e19ff, 0x761e1aff, 0x771e1aff, 0x791f1aff,
                0x7b1f1aff, 0x7d1f1bff, 0x7f1f1bff, 0x801f1bff, 0x821f1bff, 0x84201cff, 0x86201cff,
                0x88201cff, 0x89201cff, 0x8b201dff, 0x8d201dff, 0x8f201dff, 0x91211dff, 0x93211eff,
                0x94211eff, 0x96211eff, 0x98211fff, 0x9a211fff, 0x9c211fff, 0x9e211fff, 0xa02120ff,
                0xa12220ff, 0xa32220ff, 0xa52220ff, 0xa72221ff, 0xa92221ff, 0xab2221ff, 0xad2221ff,
                0xaf2222ff, 0xb12222ff, 0xb22222ff, 0xb32422ff, 0xb42622ff, 0xb52821ff, 0xb62a21ff,
                0xb72c21ff, 0xb82d21ff, 0xb92f20ff, 0xba3120ff, 0xbb3220ff, 0xbc341fff, 0xbd351fff,
                0xbe371fff, 0xbf381fff, 0xc03a1eff, 0xc13b1eff, 0xc23d1eff, 0xc33e1dff, 0xc4401dff,
                0xc5411cff, 0xc6421cff, 0xc7441cff, 0xc8451bff, 0xc9471bff, 0xca481aff, 0xcb491aff,
                0xcc4b19ff, 0xcd4c19ff, 0xce4d18ff, 0xcf4f18ff, 0xd05017ff, 0xd15217ff, 0xd25316ff,
                0xd35415ff, 0xd45515ff, 0xd55714ff, 0xd65813ff, 0xd75913ff, 0xd85b12ff, 0xd95c11ff,
                0xda5d10ff, 0xdb5f0fff, 0xdc600eff, 0xdd610dff, 0xde620cff, 0xdf640bff, 0xe06509ff,
                0xe16608ff, 0xe26807ff, 0xe36905ff, 0xe36b05ff, 0xe36d06ff, 0xe46e07ff, 0xe47007ff,
                0xe47208ff, 0xe47408ff, 0xe57609ff, 0xe5770aff, 0xe5790aff, 0xe57b0bff, 0xe57c0cff,
                0xe67e0cff, 0xe6800dff, 0xe6820eff, 0xe6830eff, 0xe6850fff, 0xe6870fff, 0xe78810ff,
                0xe78a11ff, 0xe78c11ff, 0xe78d12ff, 0xe78f13ff, 0xe79113ff, 0xe79214ff, 0xe89415ff,
                0xe89615ff, 0xe89716ff, 0xe89916ff, 0xe89a17ff, 0xe89c18ff, 0xe89e18ff, 0xe89f19ff,
                0xe8a11aff, 0xe8a21aff, 0xe9a41bff, 0xe9a61bff, 0xe9a71cff, 0xe9a91dff, 0xe9aa1dff,
                0xe9ac1eff, 0xe9ae1eff, 0xe9af1fff, 0xe9b120ff, 0xe9b220ff, 0xe9b421ff, 0xe9b522ff,
                0xe9b722ff, 0xe9b923ff, 0xe9ba23ff, 0xe9bc24ff, 0xe9bd25ff, 0xe9bf25ff, 0xe9c026ff,
                0xe9c226ff, 0xe9c327ff, 0xe9c528ff, 0xe9c728ff, 0xe9c829ff, 0xe8ca2aff, 0xe8cb2aff,
                0xe8cd2bff, 0xe8ce2bff, 0xe8d02cff, 0xe8d12dff, 0xe8d32dff, 0xe8d52eff, 0xe8d62fff,
                0xe8d82fff, 0xe7d930ff, 0xe7db30ff, 0xe7dc31ff, 0xe7de32ff, 0xe7df32ff, 0xe7e133ff,
                0xe6e234ff, 0xe6e434ff, 0xe6e535ff, 0xe7e73cff, 0xe9e745ff, 0xeae84eff, 0xece957ff,
                0xedea5eff, 0xeeeb66ff, 0xf0ec6dff, 0xf1ec75ff, 0xf2ed7cff, 0xf3ee83ff, 0xf5ef89ff,
                0xf6f090ff, 0xf7f197ff, 0xf8f19eff, 0xf9f2a4ff, 0xf9f3abff, 0xfaf4b1ff, 0xfbf5b8ff,
                0xfcf6beff, 0xfcf7c5ff, 0xfdf8cbff, 0xfdf9d2ff, 0xfef9d8ff, 0xfefadfff, 0xfefbe5ff,
                0xfffcecff, 0xfffdf2ff, 0xfffef9ff, 0xffffffff,
            ],
            Self::ExtendedKindLmann => [
                0x000000ff, 0x050004ff, 0x090009ff, 0x0d010dff, 0x100111ff, 0x130115ff, 0x160118ff,
                0x18011bff, 0x1a011eff, 0x1b0222ff, 0x1c0226ff, 0x1d022aff, 0x1d022eff, 0x1e0232ff,
                0x1e0335ff, 0x1e0339ff, 0x1e033dff, 0x1d0341ff, 0x1d0344ff, 0x1c0348ff, 0x1b044bff,
                0x1b044fff, 0x1a0452ff, 0x190455ff, 0x180458ff, 0x17045cff, 0x16055fff, 0x150562ff,
                0x140565ff, 0x130567ff, 0x12056aff, 0x12056dff, 0x11056fff, 0x0e0573ff, 0x080677ff,
                0x060878ff, 0x060b78ff, 0x060f77ff, 0x061276ff, 0x061674ff, 0x051972ff, 0x051c70ff,
                0x051f6dff, 0x05216bff, 0x052468ff, 0x052665ff, 0x052863ff, 0x052a60ff, 0x052c5eff,
                0x042e5bff, 0x043059ff, 0x043157ff, 0x043355ff, 0x043453ff, 0x043651ff, 0x04374fff,
                0x04394dff, 0x043a4cff, 0x043b4aff, 0x033d49ff, 0x033e47ff, 0x033f46ff, 0x034145ff,
                0x034243ff, 0x034342ff, 0x034441ff, 0x034540ff, 0x03473fff, 0x03483dff, 0x04493cff,
                0x044a3aff, 0x044b38ff, 0x044d37ff, 0x044e35ff, 0x044f33ff, 0x045031ff, 0x04512fff,
                0x04522dff, 0x04542bff, 0x045529ff, 0x045627ff, 0x045724ff, 0x045822ff, 0x04591fff,
                0x045b1dff, 0x045c1aff, 0x055d18ff, 0x055e15ff, 0x055f12ff, 0x05600fff, 0x05610dff,
                0x05620aff, 0x056408ff, 0x056506ff, 0x066605ff, 0x086705ff, 0x0a6805ff, 0x0b6905ff,
                0x0d6a05ff, 0x0f6b05ff, 0x116c05ff, 0x146d05ff, 0x166e05ff, 0x1a6f05ff, 0x1d7005ff,
                0x207005ff, 0x247105ff, 0x287205ff, 0x2b7306ff, 0x2f7406ff, 0x337406ff, 0x377506ff,
                0x3b7606ff, 0x3f7606ff, 0x437706ff, 0x477706ff, 0x4c7806ff, 0x507806ff, 0x547906ff,
                0x587906ff, 0x5c7a06ff, 0x617a06ff, 0x657a06ff, 0x697b06ff, 0x6d7b06ff, 0x717b06ff,
                0x767b06ff, 0x7a7b06ff, 0x7e7b06ff, 0x827b06ff, 0x877b07ff, 0x8b7b07ff, 0x907b07ff,
                0x957a07ff, 0x9a7a07ff, 0xa07908ff, 0xa57808ff, 0xab7708ff, 0xb17608ff, 0xb77509ff,
                0xbd7309ff, 0xc47109ff, 0xca6f0aff, 0xd16c0aff, 0xd8690aff, 0xde660bff, 0xe5620bff,
                0xec5e0bff, 0xf35a0cff, 0xf45b1bff, 0xf55c25ff, 0xf55e2eff, 0xf56034ff, 0xf6623aff,
                0xf6633fff, 0xf66543ff, 0xf66747ff, 0xf6694aff, 0xf66b4dff, 0xf76d4fff, 0xf76f53ff,
                0xf77057ff, 0xf7725bff, 0xf77360ff, 0xf87565ff, 0xf8766aff, 0xf87870ff, 0xf87976ff,
                0xf97a7bff, 0xf97b81ff, 0xf97d87ff, 0xf97e8dff, 0xf97f93ff, 0xf98199ff, 0xf9829eff,
                0xf983a4ff, 0xf984a9ff, 0xf985afff, 0xf986b4ff, 0xf987baff, 0xf989bfff, 0xf98ac4ff,
                0xf98bc9ff, 0xfa8cceff, 0xfa8dd3ff, 0xfa8ed8ff, 0xfa8fddff, 0xfa90e1ff, 0xfa91e6ff,
                0xfa92ebff, 0xfa93efff, 0xfa94f3ff, 0xfa95f8ff, 0xf898faff, 0xf59bfaff, 0xf29ffaff,
                0xefa2fbff, 0xeca5fbff, 0xeaa8fbff, 0xe8abfbff, 0xe6adfbff, 0xe5b0fbff, 0xe3b2fbff,
                0xe2b4fbff, 0xe1b6fbff, 0xe0b8fcff, 0xe0bafcff, 0xdfbcfcff, 0xdfbefcff, 0xdebffcff,
                0xdec1fcff, 0xdec3fcff, 0xdec4fcff, 0xdfc6fcff, 0xdfc7fcff, 0xdfc9fcff, 0xe0cafcff,
                0xe0ccfdff, 0xe1cdfdff, 0xe2cffdff, 0xe2d0fdff, 0xe3d1fdff, 0xe4d3fdff, 0xe5d4fdff,
                0xe5d5fdff, 0xe6d7fdff, 0xe7d8fdff, 0xe7dafdff, 0xe7dbfdff, 0xe8ddfdff, 0xe8defdff,
                0xe8e0fdff, 0xe8e1feff, 0xe9e3feff, 0xe9e4feff, 0xe9e6feff, 0xe9e7feff, 0xe9e9feff,
                0xe9eafeff, 0xeaecfeff, 0xeaedfeff, 0xeaeffeff, 0xebf0feff, 0xebf2feff, 0xecf3feff,
                0xedf5feff, 0xedf6feff, 0xeef7feff, 0xeff9feff, 0xf0fafeff, 0xf2fbfeff, 0xf3fcfeff,
                0xf5fdffff, 0xf8feffff, 0xfbffffff, 0xffffffff,
            ],
            Self::KindLmann => [
                0x000000ff, 0x050004ff, 0x090008ff, 0x0d010dff, 0x110110ff, 0x140114ff, 0x160117ff,
                0x19011aff, 0x1b011dff, 0x1d0220ff, 0x1e0223ff, 0x1f0226ff, 0x20022aff, 0x21022dff,
                0x220230ff, 0x230233ff, 0x240336ff, 0x250339ff, 0x25033cff, 0x26033fff, 0x260342ff,
                0x260344ff, 0x270347ff, 0x27044aff, 0x27044dff, 0x270450ff, 0x270453ff, 0x270456ff,
                0x270459ff, 0x27045dff, 0x270560ff, 0x270563ff, 0x260566ff, 0x26056aff, 0x25056dff,
                0x250570ff, 0x240674ff, 0x230677ff, 0x22067bff, 0x21067eff, 0x200681ff, 0x200684ff,
                0x1f0688ff, 0x1e078bff, 0x1d078eff, 0x1c0791ff, 0x1b0794ff, 0x1a0797ff, 0x19079aff,
                0x19079dff, 0x1808a0ff, 0x1808a3ff, 0x1408a6ff, 0x0f08aaff, 0x0809aeff, 0x080cafff,
                0x080fafff, 0x0813afff, 0x0816afff, 0x0819afff, 0x081caeff, 0x0820adff, 0x0823acff,
                0x0826aaff, 0x0829a8ff, 0x082ba6ff, 0x082ea5ff, 0x0831a3ff, 0x0833a0ff, 0x08359eff,
                0x08389cff, 0x073a9aff, 0x073c98ff, 0x073e95ff, 0x074093ff, 0x074291ff, 0x07448fff,
                0x07468dff, 0x07478bff, 0x074989ff, 0x074b87ff, 0x064c85ff, 0x064e84ff, 0x065082ff,
                0x065180ff, 0x06537fff, 0x06547dff, 0x06567bff, 0x06577aff, 0x065878ff, 0x065a77ff,
                0x065b76ff, 0x065d74ff, 0x065e73ff, 0x055f72ff, 0x066071ff, 0x056270ff, 0x05636eff,
                0x05646dff, 0x05666cff, 0x05676bff, 0x05686aff, 0x056969ff, 0x056b68ff, 0x056c67ff,
                0x056d66ff, 0x056e65ff, 0x057064ff, 0x057163ff, 0x057262ff, 0x067360ff, 0x06755fff,
                0x06765eff, 0x06775cff, 0x06785bff, 0x067a59ff, 0x067b58ff, 0x067c56ff, 0x067d54ff,
                0x067f53ff, 0x068051ff, 0x06814fff, 0x06824dff, 0x06844bff, 0x06854aff, 0x078648ff,
                0x068746ff, 0x078943ff, 0x078a41ff, 0x078b3fff, 0x078c3dff, 0x078e3bff, 0x078f38ff,
                0x079036ff, 0x079134ff, 0x079331ff, 0x07942fff, 0x07952cff, 0x07962aff, 0x079727ff,
                0x079925ff, 0x079a22ff, 0x089b1fff, 0x089c1dff, 0x089d1aff, 0x089f17ff, 0x08a014ff,
                0x08a112ff, 0x08a20fff, 0x08a30cff, 0x08a50aff, 0x08a608ff, 0x0ca708ff, 0x0fa808ff,
                0x11a908ff, 0x12aa08ff, 0x14ab08ff, 0x16ad08ff, 0x18ae08ff, 0x1aaf08ff, 0x1db008ff,
                0x20b109ff, 0x23b209ff, 0x26b309ff, 0x29b409ff, 0x2db509ff, 0x30b609ff, 0x34b709ff,
                0x38b809ff, 0x3bb909ff, 0x3fba09ff, 0x43bb09ff, 0x47bc09ff, 0x4bbd09ff, 0x4fbe09ff,
                0x53be09ff, 0x57bf09ff, 0x5bc009ff, 0x5fc109ff, 0x63c109ff, 0x67c209ff, 0x6bc309ff,
                0x6fc409ff, 0x74c409ff, 0x78c509ff, 0x7cc60aff, 0x80c60aff, 0x85c70aff, 0x89c70aff,
                0x8dc80aff, 0x91c80aff, 0x96c90aff, 0x9ac90aff, 0x9eca0aff, 0xa3ca0aff, 0xa7ca0aff,
                0xabcb0aff, 0xafcb0aff, 0xb4cb0aff, 0xb8cc0aff, 0xbccc0aff, 0xc1cc0aff, 0xc5cd0aff,
                0xc9cd0aff, 0xcdcd0aff, 0xd1cd0aff, 0xd6cd0aff, 0xdacd0bff, 0xdfcd0bff, 0xe4cd0bff,
                0xe9cd0bff, 0xedcd0bff, 0xf3cd0cff, 0xf6cc39ff, 0xf7cd56ff, 0xf8cd69ff, 0xf9ce77ff,
                0xf9cf83ff, 0xfacf8dff, 0xfad095ff, 0xfad19dff, 0xfbd2a3ff, 0xfbd3a9ff, 0xfbd4aeff,
                0xfbd6b3ff, 0xfcd7b8ff, 0xfcd8bcff, 0xfcd9c0ff, 0xfcdac3ff, 0xfcdcc7ff, 0xfcddcaff,
                0xfddecdff, 0xfde0d0ff, 0xfde1d3ff, 0xfde2d5ff, 0xfde3d8ff, 0xfde5daff, 0xfde6ddff,
                0xfde8dfff, 0xfee9e1ff, 0xfeeae3ff, 0xfeece5ff, 0xfeede7ff, 0xfeeee9ff, 0xfef0ebff,
                0xfef1edff, 0xfef2efff, 0xfef4f1ff, 0xfef5f3ff, 0xfef7f5ff, 0xfff8f6ff, 0xfff9f8ff,
                0xfffbfaff, 0xfffcfcff, 0xfffefdff, 0xffffffff,
            ],
            Self::SmoothCoolWarm => [
                0x3b4cc0ff, 0x3c4ec2ff, 0x3d50c3ff, 0x3e51c5ff, 0x3f53c7ff, 0x4055c8ff, 0x4257caff,
                0x4358cbff, 0x445accff, 0x455cceff, 0x465ecfff, 0x485fd1ff, 0x4961d2ff, 0x4a63d4ff,
                0x4b64d5ff, 0x4c66d6ff, 0x4e68d8ff, 0x4f6ad9ff, 0x506bdaff, 0x516ddbff, 0x536fddff,
                0x5470deff, 0x5572dfff, 0x5674e0ff, 0x5875e2ff, 0x5977e3ff, 0x5a78e4ff, 0x5b7ae5ff,
                0x5d7ce6ff, 0x5e7de7ff, 0x5f7fe8ff, 0x6181e9ff, 0x6282eaff, 0x6384ebff, 0x6585ecff,
                0x6687edff, 0x6788eeff, 0x698aefff, 0x6a8cf0ff, 0x6b8df0ff, 0x6d8ff1ff, 0x6e90f2ff,
                0x6f92f3ff, 0x7193f4ff, 0x7295f4ff, 0x7396f5ff, 0x7598f6ff, 0x7699f6ff, 0x779af7ff,
                0x799cf8ff, 0x7a9df8ff, 0x7b9ff9ff, 0x7da0f9ff, 0x7ea2faff, 0x80a3faff, 0x81a4fbff,
                0x82a6fbff, 0x84a7fcff, 0x85a8fcff, 0x86aafcff, 0x88abfdff, 0x89acfdff, 0x8baefdff,
                0x8caffeff, 0x8db0feff, 0x8fb1feff, 0x90b2feff, 0x92b4feff, 0x93b5ffff, 0x94b6ffff,
                0x96b7ffff, 0x97b8ffff, 0x99baffff, 0x9abbffff, 0x9bbcffff, 0x9dbdffff, 0x9ebeffff,
                0x9fbfffff, 0xa1c0ffff, 0xa2c1ffff, 0xa3c2feff, 0xa5c3feff, 0xa6c4feff, 0xa8c5feff,
                0xa9c6feff, 0xaac7fdff, 0xacc8fdff, 0xadc9fdff, 0xaec9fcff, 0xb0cafcff, 0xb1cbfcff,
                0xb2ccfbff, 0xb4cdfbff, 0xb5cefaff, 0xb6cefaff, 0xb7cff9ff, 0xb9d0f9ff, 0xbad1f8ff,
                0xbbd1f8ff, 0xbdd2f7ff, 0xbed3f6ff, 0xbfd3f6ff, 0xc0d4f5ff, 0xc1d4f4ff, 0xc3d5f4ff,
                0xc4d6f3ff, 0xc5d6f2ff, 0xc6d7f1ff, 0xc8d7f1ff, 0xc9d8f0ff, 0xcad8efff, 0xcbd8eeff,
                0xccd9edff, 0xcdd9ecff, 0xcedaebff, 0xd0daeaff, 0xd1dae9ff, 0xd2dbe8ff, 0xd3dbe7ff,
                0xd4dbe6ff, 0xd5dbe5ff, 0xd6dce4ff, 0xd7dce3ff, 0xd8dce2ff, 0xd9dce1ff, 0xdadce0ff,
                0xdbdddeff, 0xdcddddff, 0xdddcdcff, 0xdedcdbff, 0xdfdcd9ff, 0xe1dbd8ff, 0xe2dad6ff,
                0xe3dad5ff, 0xe4d9d3ff, 0xe5d9d2ff, 0xe5d8d1ff, 0xe6d8cfff, 0xe7d7ceff, 0xe8d6ccff,
                0xe9d6cbff, 0xead5c9ff, 0xebd4c8ff, 0xebd3c6ff, 0xecd3c5ff, 0xedd2c3ff, 0xeed1c2ff,
                0xeed0c0ff, 0xefcfbfff, 0xefcebdff, 0xf0cebbff, 0xf1cdbaff, 0xf1ccb8ff, 0xf2cbb7ff,
                0xf2cab5ff, 0xf3c9b4ff, 0xf3c8b2ff, 0xf4c7b1ff, 0xf4c6afff, 0xf4c5adff, 0xf5c4acff,
                0xf5c3aaff, 0xf5c1a9ff, 0xf6c0a7ff, 0xf6bfa6ff, 0xf6bea4ff, 0xf6bda2ff, 0xf7bca1ff,
                0xf7ba9fff, 0xf7b99eff, 0xf7b89cff, 0xf7b79bff, 0xf7b599ff, 0xf7b497ff, 0xf7b396ff,
                0xf7b194ff, 0xf7b093ff, 0xf7af91ff, 0xf7ad90ff, 0xf7ac8eff, 0xf7ab8cff, 0xf7a98bff,
                0xf7a889ff, 0xf7a688ff, 0xf6a586ff, 0xf6a385ff, 0xf6a283ff, 0xf6a081ff, 0xf59f80ff,
                0xf59d7eff, 0xf59c7dff, 0xf49a7bff, 0xf4997aff, 0xf49778ff, 0xf39577ff, 0xf39475ff,
                0xf29274ff, 0xf29072ff, 0xf18f71ff, 0xf18d6fff, 0xf08b6eff, 0xf08a6cff, 0xef886bff,
                0xee8669ff, 0xee8568ff, 0xed8366ff, 0xed8165ff, 0xec7f63ff, 0xeb7d62ff, 0xea7c60ff,
                0xea7a5fff, 0xe9785dff, 0xe8765cff, 0xe7745bff, 0xe67259ff, 0xe57058ff, 0xe56f56ff,
                0xe46d55ff, 0xe36b54ff, 0xe26952ff, 0xe16751ff, 0xe0654fff, 0xdf634eff, 0xde614dff,
                0xdd5f4bff, 0xdc5d4aff, 0xdb5b49ff, 0xda5947ff, 0xd85646ff, 0xd75445ff, 0xd65244ff,
                0xd55042ff, 0xd44e41ff, 0xd34c40ff, 0xd1493eff, 0xd0473dff, 0xcf453cff, 0xce433bff,
                0xcc4039ff, 0xcb3e38ff, 0xca3b37ff, 0xc83936ff, 0xc73635ff, 0xc63434ff, 0xc43132ff,
                0xc32e31ff, 0xc12b30ff, 0xc0282fff, 0xbf252eff, 0xbd222dff, 0xbc1e2cff, 0xba1a2bff,
                0xb91629ff, 0xb71128ff, 0xb60b27ff, 0xb40426ff,
            ],
        };

        Color::create_palette(&xs)
            .try_into()
            .expect("Vector length is not 256")
    }
}